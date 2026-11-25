import http.client
import json

import pytest

from arkestrator.ui.server import UiServer
from conftest import make_system


@pytest.fixture
def panel():
    system = make_system()
    ui = UiServer(system.orch, system.board, system.log, system.robot, port=0)
    ui.start()
    yield system, ui
    ui.shutdown()


def request(ui, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", ui.port, timeout=5)
    base = {"Host": f"127.0.0.1:{ui.port}"}
    base.update(headers or {})
    connection.request(method, path, body=body, headers=base)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data


def test_index_embeds_csrf_and_state_is_served(panel):
    system, ui = panel
    status, body = request(ui, "GET", "/")
    assert status == 200 and ui.csrf.encode() in body
    status, body = request(ui, "GET", "/api/state")
    payload = json.loads(body)
    assert status == 200 and payload["state"]["phase"] == "OFF"
    assert payload["health"]["backend"] == "mock"


def test_post_requires_csrf_and_local_host(panel):
    system, ui = panel
    body = json.dumps({"action": "start_session"})
    assert request(ui, "POST", "/api/action", body, {"Content-Type": "application/json"})[0] == 403
    assert request(ui, "POST", "/api/action", body, {"X-Arkestrator-CSRF": ui.csrf, "Host": "evil.example"})[0] == 403
    assert request(ui, "POST", "/api/action", body, {"X-Arkestrator-CSRF": ui.csrf, "Origin": "http://evil.example"})[0] == 403
    status, data = request(ui, "POST", "/api/action", body, {"X-Arkestrator-CSRF": ui.csrf, "Content-Type": "application/json"})
    assert status == 200 and json.loads(data)["ok"]
    system.orch.process_pending()
    assert system.orch.phase.value == "IDLE"


def test_limits_and_static_whitelist(panel):
    _, ui = panel
    big = json.dumps({"action": "command", "text": "x" * 5000})
    assert request(ui, "POST", "/api/action", big, {"X-Arkestrator-CSRF": ui.csrf})[0] == 413
    assert request(ui, "POST", "/api/voice", b"abc", {"X-Arkestrator-CSRF": ui.csrf, "Content-Type": "audio/ogg"})[0] == 415
    assert request(ui, "GET", "/static/../orchestrator.py")[0] == 404
    assert request(ui, "GET", "/static/app.js")[0] == 200
    assert request(ui, "GET", "/stream.mjpg")[0] == 404  # mock camera has no JPEG


def test_end_session_needs_token_and_both_checkboxes(panel):
    system, ui = panel
    headers = {"X-Arkestrator-CSRF": ui.csrf}
    request(ui, "POST", "/api/action", json.dumps({"action": "start_session"}), headers)
    system.orch.process_pending()
    request(ui, "POST", "/api/action", json.dumps({"action": "prepare_release"}), headers)
    system.orch.process_pending()
    token = system.orch.release_token
    status, data = request(ui, "POST", "/api/action",
                           json.dumps({"action": "end_session", "token": token, "arm_path_clear": True}), headers)
    assert status == 200 and not json.loads(data)["ok"]
    status, data = request(ui, "POST", "/api/action",
                           json.dumps({"action": "end_session", "token": token, "arm_path_clear": True,
                                       "object_supported_or_absent": True}), headers)
    assert json.loads(data)["ok"]
    system.orch.process_pending()
    assert system.orch.phase.value == "OFF"
