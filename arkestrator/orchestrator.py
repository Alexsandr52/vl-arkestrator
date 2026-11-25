"""Session daemon core: one worker thread runs sessions and fetch tasks; UI threads only post
requests and flip flags.

Task: BACKSTEP (if still at a table) → CAPTURE_COMMANDER → SEARCH → APPROACH_TABLE → GRASP
(retry) → BACKSTEP → RETURN → FIND_PERSON → APPROACH_PERSON → HANDOVER → IDLE.

* Recoverable failures (not found, grasp failed, cancelled, stalled, dropped…) end in IDLE:
  stop base → free the hand from the object → BACKSTEP if at the table → hand over what is held
  → arms to ``ready`` → say why. The next command is accepted right away.
* STOP and faults end in SAFE_HOLD. SAFE_HOLD is latched: nothing leaves it except the
  two-step release (``prepare_release`` → token → operator confirmation with that token).
"""

from __future__ import annotations

import enum
import queue
import threading
import traceback
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .errors import RecoverableError, SafeHoldError, StopRequested, TaskCancelled
from .nav.navigator import Navigator
from .profiles import TaughtPose, save_taught
from .robot.base import ReleaseConfirmation
from .skills.approach import approach_table
from .skills.common import SkillContext, measure, select_object
from .skills.grasp import grasp_object
from .skills.handover import handover
from .skills.person import approach_person, capture_commander, find_commander
from .skills.search import search_object
from .types import Task, TaskRecord
from .voice import classify_utterance, parse_fetch_keywords


class Phase(str, enum.Enum):
    OFF = "OFF"
    STARTING = "STARTING"
    IDLE = "IDLE"
    BACKSTEP = "BACKSTEP"
    CAPTURE_COMMANDER = "CAPTURE_COMMANDER"
    SEARCH = "SEARCH"
    APPROACH_TABLE = "APPROACH_TABLE"
    GRASP = "GRASP"
    RETURN = "RETURN"
    FIND_PERSON = "FIND_PERSON"
    APPROACH_PERSON = "APPROACH_PERSON"
    HANDOVER = "HANDOVER"
    RECOVERING = "RECOVERING"
    SAFE_HOLD = "SAFE_HOLD"
    RELEASE_READY = "RELEASE_READY"
    ENDING = "ENDING"
    TEACH = "TEACH"


NOT_BUSY = {Phase.OFF, Phase.IDLE, Phase.SAFE_HOLD, Phase.RELEASE_READY}
# Once in SAFE_HOLD (or the release hold), only the release path may change the phase.
LATCHED = {Phase.SAFE_HOLD, Phase.RELEASE_READY, Phase.ENDING}
LATCH_EXITS = {Phase.SAFE_HOLD, Phase.RELEASE_READY, Phase.ENDING, Phase.OFF}


def _reply(ok: bool, message: str = "", **extra) -> dict[str, Any]:
    return {"ok": ok, "message": message, **extra}


class Orchestrator:
    def __init__(
        self,
        robot,
        vision,
        perception,
        vla_pool,
        speaker,
        profiles: dict,
        cfg,
        clock,
        log,
        board,
        obstacles=None,
        taught_dir: Optional[Path] = None,
    ):
        self.robot = robot
        self.vision = vision
        self.perception = perception
        self.vla = vla_pool
        self.speaker = speaker
        self.profiles = dict(profiles)
        self.cfg = cfg
        self.clock = clock
        self.log = log
        self.board = board
        self.taught_dir = taught_dir

        self.stop_flag = threading.Event()
        self.cancel_flag = threading.Event()
        self.release_flag = threading.Event()
        self._cancel_allowed = True
        self._requests: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._pending = False
        self._shutdown = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_board_pose = -1.0

        self.phase = Phase.OFF
        self.holding: Optional[str] = None
        self.awaiting_release = False
        self.attention: Optional[str] = None
        self.release_token: Optional[str] = None
        self.current_task: Optional[Task] = None
        self._active_profile = None
        self.history: deque[dict[str, Any]] = deque(maxlen=20)

        self.nav = Navigator(robot, clock, cfg.nav, self._checkpoint, obstacles, log)
        self.ctx = SkillContext(
            robot=robot, vision=vision, nav=self.nav, vla=vla_pool, clock=clock, cfg=cfg,
            log=log, speaker=speaker, checkpoint=self._checkpoint, stage=self._stage,
        )
        self._publish()

    # ------------------------------------------------------------------ thread lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, name="orchestrator", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------------ operator API (UI threads)
    def request_session_start(self) -> dict[str, Any]:
        with self._lock:
            if self.phase != Phase.OFF or self._pending:
                return _reply(False, "Сессия уже идёт")
            self._pending = True
        self.stop_flag.clear()
        self.cancel_flag.clear()
        self.release_flag.clear()
        self._requests.put(("start", None))
        return _reply(True, "Запускаю сессию")

    def request_release_prepare(self) -> dict[str, Any]:
        """Step 1 of ending a session: freeze in the release hold and issue a one-time token."""
        with self._lock:
            if self.phase not in (Phase.IDLE, Phase.SAFE_HOLD) or self._pending:
                return _reply(False, f"Сейчас нельзя: {self.phase.value}")
            self._pending = True
        self._requests.put(("prepare_release", None))
        return _reply(True, "Готовлю отдачу рук: дождитесь удержания и подтвердите")

    def request_session_end(self, confirmation: ReleaseConfirmation) -> dict[str, Any]:
        """Step 2: operator confirmation bound to the current release token."""
        if not confirmation.valid():
            return _reply(False, "Подтвердите, что путь рук свободен и предмет поддержан или его нет")
        with self._lock:
            if self.phase != Phase.RELEASE_READY or self._pending:
                return _reply(False, "Сначала «Подготовить отдачу рук»")
            if confirmation.token != self.release_token:
                return _reply(False, "Подтверждение устарело: подготовьте отдачу рук заново")
            self._pending = True
        self._requests.put(("end", confirmation))
        return _reply(True, "Возвращаю руки и отдаю их заводскому контроллеру")

    def stop(self) -> dict[str, Any]:
        """Hard STOP (UI button only): base and arms stop now; the worker latches SAFE_HOLD."""
        self.stop_flag.set()
        try:
            self.robot.request_stop()
            self.robot.stop_base()
        finally:
            self.log.emit("operator_stop", phase=self.phase.value)
        return _reply(True, "STOP")

    def cancel(self) -> dict[str, Any]:
        if self.phase in NOT_BUSY or not self._cancel_allowed or self.awaiting_release:
            return _reply(False, "Сейчас нечего отменять")
        self.cancel_flag.set()
        self.log.emit("operator_cancel", phase=self.phase.value)
        return _reply(True, "Отменяю задачу")

    def release_object(self) -> dict[str, Any]:
        if not self.awaiting_release:
            return _reply(False, "Робот сейчас ничего не протягивает")
        self.release_flag.set()
        self.log.emit("operator_release")
        return _reply(True, "Отпускаю")

    def request_teach(self, profile_name: str) -> dict[str, Any]:
        if profile_name not in self.profiles:
            return _reply(False, f"Нет профиля {profile_name}")
        with self._lock:
            if self.phase != Phase.OFF or self._pending:
                return _reply(False, "Teach делается до начала сессии, когда руки не удерживаются")
            self._pending = True
        self._requests.put(("teach", profile_name))
        return _reply(True, "Замеряю положение предмета")

    def submit_audio(self, wav: bytes) -> dict[str, Any]:
        if self.perception is None:
            return _reply(False, "Распознавание речи недоступно")
        try:
            text = self.perception.transcribe(wav)
        except Exception as exc:
            self.log.emit("asr_failed", error=f"{type(exc).__name__}: {exc}")
            return _reply(False, "Не удалось распознать речь")
        reply = self.submit_text(text, source="voice")
        reply["heard"] = text
        return reply

    def submit_text(self, text: str, source: str = "text") -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return _reply(False, "Пустая команда")
        self.log.emit("heard", text=text, source=source)
        self.board.update(last_heard=text)
        kind = classify_utterance(text)
        if kind == "release" and self.awaiting_release:
            return self.release_object()
        if kind == "cancel":
            return self.cancel()
        if self.phase == Phase.OFF:
            return _reply(False, "Сессия не запущена: нажмите «Начать сессию»")
        if self.phase == Phase.SAFE_HOLD:
            return _reply(False, "Робот в безопасном удержании: нужно завершить сессию")
        with self._lock:
            if self.phase != Phase.IDLE or self._pending:
                return _reply(False, f"Занят ({self.phase.value}). Дождитесь окончания или отмените задачу")
            self._pending = True
        try:
            task = self._parse(text, source)
        except Exception:
            self._unreserve()
            raise
        if task is None:
            self._unreserve()
            self.ctx.say("Не понял команду")
            return _reply(False, "Не понял команду")
        ready, why = self.profiles[task.profile].ready_for_tasks
        if not ready:
            self._unreserve()
            self.ctx.say(why)
            return _reply(False, why)
        self._requests.put(("task", task))
        return _reply(True, f"Задача: {self.profiles[task.profile].accusative_ru}", task=asdict(task))

    # ------------------------------------------------------------------ worker
    def _unreserve(self) -> None:
        with self._lock:
            self._pending = False

    def _worker(self) -> None:
        while not self._shutdown.is_set():
            try:
                kind, payload = self._requests.get(timeout=0.05)
            except queue.Empty:
                self._idle_watch()
                continue
            self._dispatch(kind, payload)

    def process_pending(self) -> int:
        """Run queued requests in the calling thread (tests, scripted runs). Returns how many."""
        handled = 0
        while True:
            try:
                kind, payload = self._requests.get_nowait()
            except queue.Empty:
                return handled
            self._dispatch(kind, payload)
            handled += 1

    def _dispatch(self, kind: str, payload: Any) -> None:
        try:
            if kind == "start":
                self._do_start()
            elif kind == "task":
                if self.phase != Phase.IDLE or self.stop_flag.is_set():
                    # A fault or STOP latched SAFE_HOLD after the command was accepted.
                    self.log.emit("task_dropped", phase=self.phase.value, text=payload.text)
                else:
                    self._do_task(payload)
            elif kind == "prepare_release":
                self._do_prepare_release()
            elif kind == "end":
                self._do_end(payload)
            elif kind == "teach":
                self._do_teach(payload)
        finally:
            self._unreserve()
            self._publish()

    def _idle_watch(self) -> None:
        if self.phase == Phase.IDLE:
            try:
                if self.stop_flag.is_set():
                    raise StopRequested()
                self.robot.check()
            except SafeHoldError as err:
                self._safe_hold(err)
        self._publish()

    def _do_start(self) -> None:
        self._set_phase(Phase.STARTING)
        try:
            self.robot.start_session()
            self.ctx.arms = "ready"
            self.nav.guard = None
            self.holding = None
            self.attention = None
            self.release_token = None
            if self.nav.near_table:
                # The previous session ended at a table: the first task starts with BACKSTEP.
                self.attention = "Сессия начата у стола: первая задача начнётся с шага назад"
            self._set_phase(Phase.IDLE)
            self.ctx.say("Готов. Жду команду")
        except SafeHoldError as err:
            self._safe_hold(err)
        except Exception as exc:
            self.log.emit("session_start_failed", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
            try:
                owned = self.robot.arms_owned
            except Exception:
                owned = None  # unknown ownership is treated as owned
            if owned is False:
                self.attention = f"Сессия не запустилась: {exc}"
                self._set_phase(Phase.OFF)
            else:
                self._safe_hold(SafeHoldError("session_start_failed", error=str(exc)))

    def _do_prepare_release(self) -> None:
        try:
            self.robot.stop_base()
            token = self.robot.prepare_release()
        except Exception as exc:
            self.log.emit("prepare_release_failed", error=f"{type(exc).__name__}: {exc}")
            self.attention = f"Удержание для отдачи рук не подтверждено: {exc}"
            self._safe_hold(SafeHoldError("prepare_release_failed", error=str(exc)))
            return
        self.release_token = token
        self.log.emit("release_ready")
        self._set_phase(Phase.RELEASE_READY)

    def _do_end(self, confirmation: ReleaseConfirmation) -> None:
        if self.phase != Phase.RELEASE_READY or confirmation.token != self.release_token:
            self.log.emit("release_refused", phase=self.phase.value)
            return
        self.release_token = None  # one use, success or not
        self._set_phase(Phase.ENDING)
        try:
            self.robot.end_session(confirmation)
            self.holding = None
            self.ctx.arms = "factory"
            self._set_phase(Phase.OFF)
            self.ctx.say("Руки отданы. Сессия завершена")
        except Exception as exc:
            self.log.emit("session_end_failed", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
            self.attention = "Отдача рук не удалась: нужна новая подготовка и подтверждение"
            self._safe_hold(SafeHoldError("session_end_failed", error=str(exc)))

    def _do_teach(self, profile_name: str) -> None:
        profile = self.profiles[profile_name]
        self._set_phase(Phase.TEACH, profile=profile_name)
        try:
            labels = profile.detector_classes
            found = measure(self.ctx, labels, select_object(labels), frames_needed=5,
                            timeout_s=self.cfg.nav.measure_timeout_s * 2)
            if found is None:
                self.attention = f"Teach: {profile.display_ru} не видно"
                self.ctx.say(f"Не вижу {profile.accusative_ru}")
                return
            pose = TaughtPose(tuple(round(v, 4) for v in found.position), datetime.now().isoformat(timespec="seconds"), "teach")
            if self.taught_dir is not None:
                save_taught(Path(self.taught_dir) / f"{profile_name}.json", pose)
            self.profiles[profile_name] = replace(profile, taught=pose)
            self.log.emit("taught", profile=profile_name, object_in_robot_m=list(pose.object_in_robot_m))
            self.ctx.say(f"Запомнил, как стоять у {profile.display_ru}")
        except Exception as exc:
            self.log.emit("teach_failed", error=f"{type(exc).__name__}: {exc}")
            self.attention = f"Teach не удался: {exc}"
        finally:
            self._set_phase(Phase.OFF)

    def _do_task(self, task: Task) -> None:
        profile = self.profiles[task.profile]
        record = TaskRecord(task=task, started=self.clock.now())
        self.current_task = task
        self._active_profile = profile
        self.cancel_flag.clear()
        self.attention = None
        self.log.emit("task_started", profile=task.profile, text=task.text, attributes=list(task.attributes), source=task.source)
        try:
            self._checkpoint()
            self._task_flow(task, profile, record)
            record.outcome = "success"
        except SafeHoldError as err:
            record.outcome = f"safe_hold:{err.reason}"
            self._safe_hold(err)
        except RecoverableError as err:
            record.outcome = f"recovered:{err.reason}"
            record.grasp_attempts = err.details.get("attempts", record.grasp_attempts)
            self.log.emit("task_failed", reason=err.reason, details=err.details)
            self._recover(err, profile)
        except Exception as exc:
            record.outcome = "safe_hold:unexpected_error"
            self.log.emit("task_error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
            self._safe_hold(SafeHoldError("unexpected_error", error=f"{type(exc).__name__}: {exc}"))
        finally:
            record.finished = self.clock.now()
            summary = {
                "text": task.text,
                "profile": task.profile,
                "outcome": record.outcome,
                "grasp_attempts": record.grasp_attempts,
                "seconds": round(record.finished - record.started, 1),
                "stages": record.stages,
            }
            self.history.append(summary)
            self.log.emit("task_finished", **summary)
            self.current_task = None
            self._active_profile = None
            if self.holding is None:
                self.nav.guard = None
            if self.phase not in LATCHED:
                self._set_phase(Phase.IDLE)

    def _task_flow(self, task: Task, profile, record: TaskRecord) -> None:
        ctx = self.ctx
        nav = self.nav
        nav.begin_task()

        def stage(phase: Phase, **details) -> None:
            record.stages.append(phase.value)
            self._stage(phase.value, **details)

        if nav.near_table:
            stage(Phase.BACKSTEP)
            nav.backstep()
            nav.begin_task()
        self.vla.prefetch(profile)

        stage(Phase.CAPTURE_COMMANDER)
        commander = capture_commander(ctx)
        self.board.update(commander=None if commander.world_xy is None else list(commander.world_xy))

        stage(Phase.SEARCH)
        fix = search_object(ctx, task, profile)
        self.board.update(object_fix=list(fix.world_xy))

        stage(Phase.APPROACH_TABLE)
        fix, error = approach_table(ctx, profile, fix)

        stage(Phase.GRASP)
        result = grasp_object(ctx, profile, fix)
        record.grasp_attempts = result.attempts
        self.holding = profile.name
        if profile.grasp.carry_check != "none":
            nav.guard = lambda: self.robot.carry_check(profile)
        ctx.say(f"Взял {profile.accusative_ru}")

        stage(Phase.BACKSTEP)
        nav.backstep()

        stage(Phase.RETURN)
        nav.retrace()

        stage(Phase.FIND_PERSON)
        person = find_commander(ctx, commander)
        if person is not None:
            stage(Phase.APPROACH_PERSON)
            approach_person(ctx, commander, person)
        else:
            self.log.emit("commander_not_found")
            ctx.say("Не нашёл вас. Заберите, пожалуйста, здесь")

        stage(Phase.HANDOVER)
        nav.guard = None
        handover(ctx, profile, self.release_flag, self._set_awaiting_release)
        self.holding = None
        ctx.say("Готово. Жду следующую команду")

    def _recover(self, err: RecoverableError, profile) -> None:
        self._set_phase(Phase.RECOVERING, reason=err.reason)
        self._cancel_allowed = False
        self.cancel_flag.clear()
        try:
            self.robot.stop_base()
            self.nav.wait_still()  # arms move only once the base is measured still
            if err.reason == "object_dropped":
                self.holding = None
                self.nav.guard = None
            if err.say:
                self.ctx.say(err.say)
            if self.ctx.arms == "carry_unverified":
                # Lifted but not verified: put it back down where it was, then open.
                self.robot.release_at_table(profile)
                self.ctx.arms = "grasp"
                self.robot.retract_after_miss(profile)
                self.ctx.arms = "grasp_retracted"
            if self.ctx.arms == "grasp":
                # The open hand may still be at the object: open, lift away, then tuck.
                self.robot.open_hands(profile.grasp.hand)
                self.robot.retract_after_miss(profile)
                self.ctx.arms = "grasp_retracted"
            if self.ctx.arms == "grasp_retracted":
                # Tuck the arms before stepping back from the table.
                self.robot.move_arms(profile.ready_pose)
                self.ctx.arms = "ready"
            if self.nav.near_table:
                self._stage(Phase.BACKSTEP.value, during="recovery")
                self.nav.backstep()
            if self.holding is not None:
                self.nav.guard = None
                handover(self.ctx, profile, self.release_flag, self._set_awaiting_release)
                self.holding = None
            elif self.ctx.arms != "ready":
                self.robot.move_arms(profile.ready_pose)
                self.ctx.arms = "ready"
            self.log.emit("recovered", reason=err.reason)
        except SafeHoldError as hold:
            self._safe_hold(hold)
        except RecoverableError as again:
            # Arms/object state is no longer known to be consistent: no new tasks until release.
            self.attention = f"Восстановление не завершено: {again.reason}"
            self.log.emit("recovery_incomplete", reason=again.reason, details=again.details)
            self._safe_hold(SafeHoldError(f"recovery_incomplete:{again.reason}"))
        except Exception as exc:
            self.log.emit("recovery_error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
            self._safe_hold(SafeHoldError("unexpected_error_in_recovery", error=str(exc)))
        finally:
            self._cancel_allowed = True
            self.cancel_flag.clear()

    def _safe_hold(self, err: SafeHoldError) -> None:
        self.cancel_flag.clear()
        self.awaiting_release = False
        try:
            self.robot.enter_safe_hold(err.reason)
        except Exception as exc:
            self.log.emit("safe_hold_error", error=f"{type(exc).__name__}: {exc}")
            self.attention = f"Удержание НЕ подтверждено бэкендом: {exc}"
        self.log.emit("safe_hold", reason=err.reason, details=err.details)
        self._set_phase(Phase.SAFE_HOLD, reason=err.reason)

    # ------------------------------------------------------------------ helpers
    def _checkpoint(self) -> None:
        if self.stop_flag.is_set():
            raise StopRequested()
        self.robot.check()
        if self._cancel_allowed and self.cancel_flag.is_set():
            self.cancel_flag.clear()
            raise TaskCancelled()
        now = self.clock.now()
        if now - self._last_board_pose >= 0.2:
            self._last_board_pose = now
            self._publish()

    def _set_awaiting_release(self, waiting: bool) -> None:
        self.awaiting_release = waiting
        self._publish()

    def _set_phase(self, phase: Phase, **details) -> None:
        if self.phase in LATCHED and phase not in LATCH_EXITS:
            self.log.emit("phase_change_refused", current=self.phase.value, requested=phase.value)
            return
        self.phase = phase
        self.board.update(phase=phase.value, stage_details=details)
        self.log.emit("phase", phase=phase.value, **details)
        self._publish()

    def _stage(self, name: str, **details) -> None:
        phase = Phase.__members__.get(name)
        if phase is not None and phase != self.phase:
            self._set_phase(phase, **details)
        else:
            self.board.update(stage_details=details)
            self.log.emit("stage", stage=name, **details)

    def _parse(self, text: str, source: str) -> Optional[Task]:
        catalog = [p.catalog_entry() for p in self.profiles.values()]
        parsed = None
        parser = "vlm"
        if self.perception is not None:
            try:
                parsed = self.perception.parse_command(text, catalog)
                if parsed is None:
                    self.log.emit("parse", parser="vlm", result=None)
                    return None
            except Exception as exc:
                self.log.emit("parse_vlm_failed", error=f"{type(exc).__name__}: {exc}")
                parsed = None
        if parsed is None:
            parser = "keywords"
            parsed = parse_fetch_keywords(text, catalog)
        self.log.emit("parse", parser=parser, result=parsed)
        if not parsed or parsed.get("profile") not in self.profiles:
            return None
        profile = self.profiles[parsed["profile"]]
        return Task(
            profile=profile.name,
            object_label=str(parsed.get("object") or profile.display_ru),
            text=text,
            source=f"{source}/{parser}",
            attributes=tuple(str(a) for a in parsed.get("attributes") or ()),
        )

    def allowed_actions(self) -> list[str]:
        phase = self.phase
        if phase == Phase.OFF:
            return ["start_session", "teach"]
        if phase == Phase.IDLE:
            return ["command", "stop", "prepare_release"]
        if phase == Phase.SAFE_HOLD:
            return ["prepare_release"]
        if phase == Phase.RELEASE_READY:
            return ["end_session"]
        if phase in (Phase.ENDING, Phase.TEACH):
            return []
        actions = ["stop"]
        if self.awaiting_release:
            actions.append("release")
        elif self._cancel_allowed:
            actions.append("cancel")
        return actions

    def _publish(self) -> None:
        try:
            odom = self.robot.odometry()
            pose = [round(v, 3) for v in odom.pose.as_list()]
        except Exception:
            pose = None
        self.board.update(
            phase=self.phase.value,
            allowed=self.allowed_actions(),
            session_active=bool(self.robot.arms_owned),
            task=None if self.current_task is None else asdict(self.current_task),
            holding=self.holding,
            awaiting_release=self.awaiting_release,
            near_table=self.nav.near_table,
            pose=pose,
            waypoints=[[round(v, 3) for v in p.as_list()] for p in self.nav.waypoints],
            attention=self.attention,
            release_token=self.release_token,
            said=self.speaker.recent()[-8:],
            history=list(self.history),
            vla=self.vla.status() if hasattr(self.vla, "status") else {},
            profiles={
                name: {
                    "display": p.display_ru,
                    "ready": p.ready_for_tasks[0],
                    "why_not": p.ready_for_tasks[1],
                    "taught": None if p.taught is None else list(p.taught.object_in_robot_m),
                }
                for name, p in self.profiles.items()
            },
        )
