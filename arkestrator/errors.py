"""Error classes decide where a failed task goes.

* ``RecoverableError``: the task failed but the robot is healthy. The orchestrator stops the
  base, steps back from the table if needed, returns the arms to ``ready`` and waits for the
  next command (IDLE).
* ``SafeHoldError``: STOP, overheating, mode change, worker or camera fault, or any unknown
  exception. Arms are held; only the existing operator-confirmed release ends the session.
"""

from __future__ import annotations


class ArkestratorError(Exception):
    def __init__(self, reason: str, say: str | None = None, **details):
        super().__init__(reason if not details else f"{reason}: {details}")
        self.reason = reason
        self.say = say
        self.details = details


class RecoverableError(ArkestratorError):
    pass


class TaskCancelled(RecoverableError):
    def __init__(self, say: str | None = "Отменяю задачу"):
        super().__init__("cancelled", say=say)


class SafeHoldError(ArkestratorError):
    pass


class StopRequested(SafeHoldError):
    def __init__(self):
        super().__init__("stop_requested", say=None)


class NavigationRuleError(RuntimeError):
    """A skill broke a navigation invariant (e.g. turning at the table without BACKSTEP).

    This is a programming error, so it is not recoverable: it ends in SAFE_HOLD.
    """
