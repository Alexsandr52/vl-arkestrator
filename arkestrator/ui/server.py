"""Local web panel (stdlib HTTP, pattern of GR00T-pickup-bottle ``tools/bottle_control_ui.py``).

Binds to 127.0.0.1 only; from the laptop use ``ssh -L 8770:127.0.0.1:8770 g1-gpu``, which also
gives the browser a secure (localhost) context for the microphone. Every POST needs the per-launch
``X-Arkestrator-CSRF`` token; Host and Origin must be this panel.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from ..robot.base import ReleaseConfirmation

STATIC = Path(__file__).with_name("static")
STATIC_FILES = {
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "recorder-worklet.js": "text/javascript; charset=utf-8",
}
MAX_JSON_BYTES = 4096
MAX_AUDIO_BYTES = 4 * 1024 * 1024


class UiServer:
    def __init__(self, orchestrator, board, log, robot, host: str = "127.0.0.1", port: int = 8770):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("The panel binds to localhost only; use an SSH tunnel")
        self.orchestrator = orchestrator
        self.board = board
        self.log = log
        self.robot = robot
        self.csrf = secrets.token_urlsafe(24)
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="ui", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # ------------------------------------------------------------------ actions
    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        orch = self.orchestrator
        action = payload.get("action")
        if action == "start_session":
            return orch.request_session_start()
        if action == "stop":
            return orch.stop()
        if action == "cancel":
            return orch.cancel()
        if action == "release":
            return orch.release_object()
        if action == "prepare_release":
            return orch.request_release_prepare()
        if action == "end_session":
            confirmation = ReleaseConfirmation(
                token=str(payload.get("token") or ""),
                arm_path_clear=payload.get("arm_path_clear") is True,
                object_supported_or_absent=payload.get("object_supported_or_absent") is True,
            )
            return orch.request_session_end(confirmation)
        if action == "command":
            text = payload.get("text")
            if not isinstance(text, str) or len(text) > 300:
                return {"ok": False, "message": "Команда должна быть строкой до 300 символов"}
            return orch.submit_text(text, source="text")
        if action == "teach":
            return orch.request_teach(str(payload.get("profile", "")))
        return {"ok": False, "message": f"Неизвестное действие {action!r}"}

    def state(self, after: int) -> dict[str, Any]:
        try:
            health = self.robot.health()
        except Exception as exc:
            health = {"error": f"{type(exc).__name__}: {exc}"}
        return {"state": self.board.snapshot(), "health": health, "events": self.log.tail(after)}

    def latest_jpeg(self) -> Optional[bytes]:
        try:
            frame = self.robot.frames().get("head")
        except Exception:
            return None
        return None if frame is None else frame.jpeg

    # ------------------------------------------------------------------ HTTP
    def _handler(self):
        ui = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "arkestrator"
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # keep the daemon log clean
                pass

            def _allowed_origin(self) -> bool:
                allowed = {f"127.0.0.1:{ui.port}", f"localhost:{ui.port}"}
                if self.headers.get("Host") not in allowed:
                    return False
                origin = self.headers.get("Origin")
                return origin is None or origin in {f"http://{h}" for h in allowed}

            def _send(self, status: int, body: bytes, content_type: str, extra: Optional[dict] = None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                for key, value in (extra or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, value: Any, extra: Optional[dict] = None):
                self._send(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", extra)

            def _read_body(self, limit: int) -> Optional[bytes]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if length < 0 or length > limit:
                    # The body stays unread: close, or its bytes would be parsed as the next request.
                    self.close_connection = True
                    self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "message": "Слишком большой запрос"},
                               {"Connection": "close"})
                    return None
                return self.rfile.read(length)

            def do_GET(self):
                if not self._allowed_origin():
                    return self._json(HTTPStatus.FORBIDDEN, {"ok": False, "message": "forbidden"})
                url = urlparse(self.path)
                if url.path == "/":
                    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("{{CSRF}}", ui.csrf)
                    return self._send(
                        HTTPStatus.OK, html.encode(), "text/html; charset=utf-8",
                        {"Content-Security-Policy": "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'"},
                    )
                if url.path.startswith("/static/"):
                    name = url.path[len("/static/"):]
                    if name not in STATIC_FILES:
                        return self._json(HTTPStatus.NOT_FOUND, {"ok": False})
                    return self._send(HTTPStatus.OK, (STATIC / name).read_bytes(), STATIC_FILES[name])
                if url.path == "/api/state":
                    try:
                        after = int(parse_qs(url.query).get("after", ["0"])[0])
                    except ValueError:
                        after = 0
                    return self._json(HTTPStatus.OK, ui.state(after))
                if url.path == "/api/camera.jpg":
                    jpeg = ui.latest_jpeg()
                    if jpeg is None:
                        return self._json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "нет кадра"})
                    return self._send(HTTPStatus.OK, jpeg, "image/jpeg")
                if url.path == "/stream.mjpg":
                    return self._mjpeg()
                return self._json(HTTPStatus.NOT_FOUND, {"ok": False})

            def _mjpeg(self):
                if ui.latest_jpeg() is None:
                    return self._json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "нет камеры"})
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    while True:
                        jpeg = ui.latest_jpeg()
                        if jpeg is not None:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                            )
                            self.wfile.flush()
                        time.sleep(0.2)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self):
                # Some answers (403/413/415) leave the body unread: never reuse the connection.
                self.close_connection = True
                if not self._allowed_origin():
                    return self._json(HTTPStatus.FORBIDDEN, {"ok": False, "message": "forbidden"})
                if not secrets.compare_digest(self.headers.get("X-Arkestrator-CSRF", ""), ui.csrf):
                    return self._json(HTTPStatus.FORBIDDEN, {"ok": False, "message": "csrf"})
                url = urlparse(self.path)
                if url.path == "/api/action":
                    body = self._read_body(MAX_JSON_BYTES)
                    if body is None:
                        return
                    try:
                        payload = json.loads(body or b"{}")
                        if not isinstance(payload, dict):
                            raise ValueError
                    except ValueError:
                        return self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": "bad json"})
                    try:
                        result = ui.action(payload)
                    except Exception as exc:
                        ui.log.emit("ui_action_error", action=payload.get("action"), error=f"{type(exc).__name__}: {exc}")
                        return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "message": str(exc)})
                    return self._json(HTTPStatus.OK, result)
                if url.path == "/api/voice":
                    if not self.headers.get("Content-Type", "").startswith("audio/wav"):
                        return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"ok": False, "message": "нужен audio/wav"})
                    body = self._read_body(MAX_AUDIO_BYTES)
                    if body is None:
                        return
                    return self._json(HTTPStatus.OK, ui.orchestrator.submit_audio(body))
                return self._json(HTTPStatus.NOT_FOUND, {"ok": False})

        return Handler
