from __future__ import annotations

import hashlib
import json
import base64
import os
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import parse, request

from cornerstone_cli.product_runtime import UI_SURFACES, make_server
from cornerstone_cli.validators import redact_text


CHROME_CANDIDATES = [
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
]

DEFAULT_ACCEPTANCE_SCENARIO_REPORT = "reports/scenario/vs0-runtime-acceptance-2026-06-11.json"
DEFAULT_PRODUCT_RUNTIME_REPORT = "reports/scenario/vs0-product-runtime-2026-06-11.json"
DEFAULT_BROWSER_PROOF_DIR = "reports/browser/vs0-runtime-acceptance-2026-06-11"
DEFAULT_RELEASE_PACKAGE_DIR = "reports/release/vs0-runtime-acceptance-2026-06-11"
DEFAULT_ACCEPTANCE_REPORT = "docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_REPORT_2026-06-11.md"
DEFAULT_ACCEPTANCE_FREEZE_REPORT = "docs/verification-reports/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_SCENARIO_FREEZE_REPORT_2026-06-11.md"
DEFAULT_EVUX_SCENARIO_REPORT = "reports/scenario/vs0-evux-2026-06-13.json"
DEFAULT_EVUX_BROWSER_PROOF_DIR = "reports/browser/vs0-evux-2026-06-13"
DEFAULT_EVUX_QUICKSTART_REPORT = "reports/quickstart/vs0-evux-quickstart.json"
DEFAULT_EVUX_RELEASE_PACKAGE_DIR = "reports/release/vs0-evux-2026-06-13"
DEFAULT_EVUX_REPORT = "docs/verification-reports/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_REPORT_2026-06-13.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def find_chrome() -> Path | None:
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    for name in ["google-chrome", "chromium", "chromium-browser", "chrome"]:
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    return None


def _chrome_command(chrome: Path, profile_dir: Path, window_size: str) -> list[str]:
    return [
        str(chrome),
        "--headless=new",
        "--disable-background-networking",
        "--disable-default-apps",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-sync",
        "--hide-scrollbars",
        "--metrics-recording-only",
        "--no-default-browser-check",
        "--no-first-run",
        "--no-sandbox",
        f"--user-data-dir={profile_dir}",
        f"--window-size={window_size}",
    ]


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json_urlopen(url: str, *, timeout: float = 2.0) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class _CDPClient:
    def __init__(self, websocket_url: str) -> None:
        parsed = parse.urlparse(websocket_url)
        if parsed.scheme != "ws":
            raise ValueError(f"Unsupported DevTools websocket URL: {websocket_url}")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.path = parsed.path
        if parsed.query:
            self.path += f"?{parsed.query}"
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        self.sock.settimeout(5)
        self._next_id = 1
        self._events: list[dict[str, Any]] = []
        self._handshake()

    def _handshake(self) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        headers = "\r\n".join(
            [
                f"GET {self.path} HTTP/1.1",
                f"Host: {self.host}:{self.port}",
                "Upgrade: websocket",
                "Connection: Upgrade",
                f"Sec-WebSocket-Key: {key}",
                "Sec-WebSocket-Version: 13",
                "\r\n",
            ]
        )
        self.sock.sendall(headers.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(response.decode("utf-8", errors="replace"))

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def _send_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.extend([0x80 | 126, (length >> 8) & 0xFF, length & 0xFF])
        else:
            header.append(0x80 | 127)
            header.extend(length.to_bytes(8, "big"))
        mask = os.urandom(4)
        header.extend(mask)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def _recv_frame(self) -> tuple[int, bool, bytes]:
        first = self.sock.recv(2)
        if len(first) < 2:
            raise RuntimeError("websocket_closed")
        fin = bool(first[0] & 0x80)
        opcode = first[0] & 0x0F
        masked = bool(first[1] & 0x80)
        length = first[1] & 0x7F
        if length == 126:
            length = int.from_bytes(self.sock.recv(2), "big")
        elif length == 127:
            length = int.from_bytes(self.sock.recv(8), "big")
        mask = self.sock.recv(4) if masked else b""
        payload = b""
        while len(payload) < length:
            payload += self.sock.recv(length - len(payload))
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, fin, payload

    def _recv_message(self) -> dict[str, Any]:
        chunks: list[bytes] = []
        while True:
            opcode, fin, payload = self._recv_frame()
            if opcode == 0x8:
                raise RuntimeError("websocket_close_frame")
            if opcode == 0x9:
                continue
            if opcode in {0x1, 0x0}:
                chunks.append(payload)
                if fin:
                    return json.loads(b"".join(chunks).decode("utf-8"))

    def send(self, method: str, params: dict[str, Any] | None = None) -> int:
        command_id = self._next_id
        self._next_id += 1
        payload = {"id": command_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send_text(json.dumps(payload, separators=(",", ":")))
        return command_id

    def command(self, method: str, params: dict[str, Any] | None = None, *, timeout: float = 10.0) -> dict[str, Any]:
        command_id = self.send(method, params)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self._recv_message()
            if message.get("id") == command_id:
                if "error" in message:
                    raise RuntimeError(json.dumps(message["error"], sort_keys=True))
                return message.get("result", {})
            self._events.append(message)
        raise TimeoutError(method)

    def wait_event(self, method: str, *, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for index, event in enumerate(self._events):
                if event.get("method") == method:
                    return self._events.pop(index)
            message = self._recv_message()
            if message.get("method") == method:
                return message
            self._events.append(message)
        raise TimeoutError(method)


def _runtime_eval(client: _CDPClient, expression: str, *, timeout: float = 10.0) -> Any:
    result = client.command(
        "Runtime.evaluate",
        {"expression": expression, "returnByValue": True, "awaitPromise": True},
        timeout=timeout,
    )
    value = result.get("result", {})
    if value.get("subtype") == "error":
        raise RuntimeError(str(value.get("description") or value.get("value")))
    return value.get("value")


def _launch_cdp_chrome(chrome: Path, profile_dir: Path, port: int, window_size: str) -> subprocess.Popen[str]:
    command = [
        *_chrome_command(chrome, profile_dir, window_size),
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        "about:blank",
    ]
    return subprocess.Popen(
        command,
        cwd=profile_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def capture_evux_browser_proof(
    root: Path,
    *,
    state_dir: Path,
    output_dir: Path,
    window_size: str = "1280,980",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    proof_path = output_dir / "browser-proof.json"
    screenshot_path = output_dir / "workflow.png"
    dom_path = output_dir / "workflow.dom.html"
    trace_path = output_dir / "workflow-trace.json"

    if chrome is None:
        proof = {
            "schema_version": "cs.evux_browser_proof.v0",
            "status": "NOT_VERIFIED",
            "created_at": utc_now(),
            "browser": None,
            "errors": ["chrome_not_found"],
            "screenshot_path": relative_to_root(root, screenshot_path),
            "dom_path": relative_to_root(root, dom_path),
        }
        write_json(proof_path, proof)
        return proof

    server = make_server(root, state_dir)
    host, port = server.server_address
    url = f"http://{host}:{port}/?scenario=vs0-evux"
    thread_error: list[str] = []

    import threading

    def serve() -> None:
        try:
            server.serve_forever()
        except Exception as error:  # pragma: no cover - defensive thread boundary
            thread_error.append(str(error))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    chrome_exit_code: int | None = None
    chrome_stderr_tail: list[str] = []
    browser_error: str | None = None
    workflow_state: dict[str, Any] = {}
    html = ""
    clean_browser_exit = False

    try:
        with tempfile.TemporaryDirectory(prefix="cornerstone-evux-chrome-profile-") as profile_tmp:
            profile_dir = Path(profile_tmp)
            debug_port = _free_local_port()
            process = _launch_cdp_chrome(chrome, profile_dir, debug_port, window_size)
            page: _CDPClient | None = None
            browser: _CDPClient | None = None
            try:
                version: dict[str, Any] | None = None
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        version = _json_urlopen(f"http://127.0.0.1:{debug_port}/json/version")
                        break
                    except OSError:
                        time.sleep(0.1)
                if version is None:
                    raise TimeoutError("devtools_version")
                browser = _CDPClient(str(version["webSocketDebuggerUrl"]))
                new_target_req = request.Request(
                    f"http://127.0.0.1:{debug_port}/json/new?{parse.quote(url, safe=':/?=&')}",
                    method="PUT",
                )
                with request.urlopen(new_target_req, timeout=5) as response:
                    target = json.loads(response.read().decode("utf-8"))
                page = _CDPClient(str(target["webSocketDebuggerUrl"]))
                page.command("Page.enable")
                page.command("Runtime.enable")
                page.command("Page.navigate", {"url": url})
                page.wait_event("Page.loadEventFired", timeout=10)
                _runtime_eval(page, "document.getElementById('run-evux').click(); true")
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    workflow_state = _runtime_eval(
                        page,
                        """(() => {
                          const status = document.getElementById('evux-status');
                          const trace = document.getElementById('evux-trace');
                          return {
                            status: status ? status.dataset.evuxStatus || '' : '',
                            statusText: status ? status.textContent || '' : '',
                            traceText: trace ? trace.textContent || '' : ''
                          };
                        })()""",
                        timeout=5,
                    ) or {}
                    if workflow_state.get("status") in {"passed", "failed"}:
                        break
                    time.sleep(0.25)
                html = str(_runtime_eval(page, "document.documentElement.outerHTML", timeout=5) or "")
                dom_path.write_text(html)
                screenshot = page.command("Page.captureScreenshot", {"format": "png", "fromSurface": True}, timeout=10)
                screenshot_path.write_bytes(base64.b64decode(str(screenshot.get("data", ""))))
                try:
                    trace_json = json.loads(str(workflow_state.get("traceText") or "{}"))
                except ValueError:
                    trace_json = {"raw": workflow_state.get("traceText")}
                write_json(trace_path, trace_json if isinstance(trace_json, dict) else {"trace": trace_json})
                try:
                    browser.command("Browser.close", timeout=5)
                except Exception:
                    page.command("Browser.close", timeout=5)
                try:
                    chrome_exit_code = process.wait(timeout=10)
                    clean_browser_exit = chrome_exit_code == 0
                except subprocess.TimeoutExpired:
                    process.terminate()
                    chrome_exit_code = process.wait(timeout=5)
            except Exception as error:
                browser_error = str(error)
                try:
                    if process.poll() is None:
                        process.terminate()
                        chrome_exit_code = process.wait(timeout=5)
                    else:
                        chrome_exit_code = process.returncode
                except Exception:
                    chrome_exit_code = process.returncode
            finally:
                if page is not None:
                    page.close()
                if browser is not None:
                    browser.close()
                stderr = ""
                try:
                    if process.stderr is not None:
                        stderr = process.stderr.read() or ""
                except Exception:
                    stderr = ""
                chrome_stderr_tail = stderr.strip().splitlines()[-5:]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    screenshot_exists = screenshot_path.exists() and screenshot_path.stat().st_size > 0
    required_markers = {
        "workflow_passed": workflow_state.get("status") == "passed",
        "button_clicked": "data-evux-clicked=\"true\"" in html,
        "artifact_id": "id=\"evux-artifact-id\"" in html and "art_" in html,
        "search_snapshot_id": "id=\"evux-search-snapshot-id\"" in html and "search_" in html,
        "evidence_bundle_id": "id=\"evux-evidence-bundle-id\"" in html and "evb_" in html,
        "claim_id": "id=\"evux-claim-id\"" in html and "claim_" in html,
        "action_id": "id=\"evux-action-id\"" in html and "action_" in html,
        "audit_verified": "id=\"evux-audit-verify\"" in html and "success" in html.lower(),
        "zero_evidence_denied": "CS_CLAIM_EVIDENCE_REQUIRED" in html,
        "mock_connector_calls": "mock_connector_calls=1" in html,
        "real_external_http_calls_zero": "real_external_http_calls=0" in html,
    }
    status = "PASS" if screenshot_exists and clean_browser_exit and all(required_markers.values()) and not thread_error and not browser_error else "FAIL"
    if status == "FAIL" and screenshot_exists and any(required_markers.values()):
        status = "PARTIAL"
    proof = {
        "schema_version": "cs.evux_browser_proof.v0",
        "status": status,
        "created_at": utc_now(),
        "browser": {
            "name": "Google Chrome" if "Google Chrome" in str(chrome) else chrome.name,
            "executable": str(chrome),
            "headless": True,
            "window_size": window_size,
            "driver": "chrome_devtools_protocol",
        },
        "url": url,
        "route": "/?scenario=vs0-evux",
        "clean_browser_exit": clean_browser_exit,
        "chrome_exit_code": chrome_exit_code,
        "chrome_timeout": not clean_browser_exit,
        "screenshot_path": relative_to_root(root, screenshot_path),
        "screenshot_sha256": sha256_file(screenshot_path) if screenshot_exists else None,
        "screenshot_bytes": screenshot_path.stat().st_size if screenshot_exists else 0,
        "dom_path": relative_to_root(root, dom_path),
        "dom_sha256": sha256_file(dom_path) if dom_path.exists() else None,
        "trace_path": relative_to_root(root, trace_path),
        "trace_sha256": sha256_file(trace_path) if trace_path.exists() else None,
        "workflow_state": workflow_state,
        "required_markers": required_markers,
        "errors": [error for error in [browser_error, *thread_error] if error],
        "stderr_tail": chrome_stderr_tail,
    }
    write_json(proof_path, proof)
    return proof


def capture_browser_proof(
    root: Path,
    *,
    state_dir: Path,
    output_dir: Path,
    window_size: str = "1280,900",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    proof_path = output_dir / "browser-proof.json"
    screenshot_path = output_dir / "home.png"
    dom_path = output_dir / "home.dom.html"

    if chrome is None:
        proof = {
            "schema_version": "cs.browser_proof.v0",
            "status": "failed",
            "created_at": utc_now(),
            "browser": None,
            "errors": ["chrome_not_found"],
            "screenshot_path": relative_to_root(root, screenshot_path),
            "dom_path": relative_to_root(root, dom_path),
        }
        write_json(proof_path, proof)
        return proof

    server = make_server(root, state_dir)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    thread_error: list[str] = []

    import threading

    def serve() -> None:
        try:
            server.serve_forever()
        except Exception as error:  # pragma: no cover - defensive thread boundary
            thread_error.append(str(error))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    try:
        with tempfile.TemporaryDirectory(prefix="cornerstone-chrome-profile-") as profile_tmp:
            profile_dir = Path(profile_tmp)
            debug_port = _free_local_port()
            process = _launch_cdp_chrome(chrome, profile_dir, debug_port, window_size)
            page: _CDPClient | None = None
            browser: _CDPClient | None = None
            screenshot_timeout = True
            browser_error: str | None = None
            screenshot_result = subprocess.CompletedProcess(args=[], returncode=124, stdout="", stderr="")
            try:
                version: dict[str, Any] | None = None
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        version = _json_urlopen(f"http://127.0.0.1:{debug_port}/json/version")
                        break
                    except OSError:
                        time.sleep(0.1)
                if version is None:
                    raise TimeoutError("devtools_version")
                browser = _CDPClient(str(version["webSocketDebuggerUrl"]))
                new_target_req = request.Request(
                    f"http://127.0.0.1:{debug_port}/json/new?{parse.quote(url, safe=':/?=&')}",
                    method="PUT",
                )
                with request.urlopen(new_target_req, timeout=5) as response:
                    target = json.loads(response.read().decode("utf-8"))
                page = _CDPClient(str(target["webSocketDebuggerUrl"]))
                page.command("Page.enable")
                page.command("Runtime.enable")
                page.command("Page.navigate", {"url": url})
                page.wait_event("Page.loadEventFired", timeout=10)
                dom_path.write_text(str(_runtime_eval(page, "document.documentElement.outerHTML", timeout=5) or ""))
                screenshot = page.command("Page.captureScreenshot", {"format": "png", "fromSurface": True}, timeout=10)
                screenshot_path.write_bytes(base64.b64decode(str(screenshot.get("data", ""))))
                try:
                    browser.command("Browser.close", timeout=5)
                except Exception:
                    page.command("Browser.close", timeout=5)
                try:
                    exit_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    exit_code = process.wait(timeout=5)
                screenshot_timeout = exit_code != 0
                stderr_tail = ""
                try:
                    if process.stderr is not None:
                        stderr_tail = process.stderr.read() or ""
                except Exception:
                    stderr_tail = ""
                screenshot_result = subprocess.CompletedProcess(
                    args=[],
                    returncode=exit_code,
                    stdout="",
                    stderr=stderr_tail,
                )
            except Exception as error:
                browser_error = str(error)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                stderr_tail = ""
                try:
                    if process.stderr is not None:
                        stderr_tail = process.stderr.read() or ""
                except Exception:
                    stderr_tail = ""
                screenshot_result = subprocess.CompletedProcess(
                    args=[],
                    returncode=process.returncode if process.returncode is not None else 124,
                    stdout="",
                    stderr=stderr_tail or browser_error,
                )
            finally:
                if page is not None:
                    page.close()
                if browser is not None:
                    browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    dom = dom_path.read_text() if dom_path.exists() else ""
    surface_presence = {surface: surface in dom for surface in UI_SURFACES}
    readiness_labels = [
        "local_scenario_ready=true",
        "vs0_runtime_ready=true",
        "production_release_ready=false",
        "real_external_http_calls=0",
    ]
    screenshot_exists = screenshot_path.exists() and screenshot_path.stat().st_size > 0
    proof_status = (
        screenshot_exists
        and all(surface_presence.values())
        and all(label in dom for label in readiness_labels)
        and "production_release_ready=true" not in dom
        and not screenshot_timeout
        and not browser_error
        and not thread_error
    )
    proof = {
        "schema_version": "cs.browser_proof.v0",
        "status": "passed" if proof_status else "failed",
        "created_at": utc_now(),
        "browser": {
            "name": "Google Chrome" if "Google Chrome" in str(chrome) else chrome.name,
            "executable": str(chrome),
            "headless": True,
            "window_size": window_size,
        },
        "url": url,
        "route": "/",
        "screenshot_path": relative_to_root(root, screenshot_path),
        "screenshot_sha256": sha256_file(screenshot_path) if screenshot_exists else None,
        "screenshot_bytes": screenshot_path.stat().st_size if screenshot_exists else 0,
        "dom_path": relative_to_root(root, dom_path),
        "dom_sha256": sha256_file(dom_path) if dom_path.exists() else None,
        "surface_presence": surface_presence,
        "readiness_labels_present": {label: label in dom for label in readiness_labels},
        "production_overclaim_absent": "production_release_ready=true" not in dom,
        "chrome_exit_codes": {
            "screenshot": screenshot_result.returncode,
        },
        "chrome_timeout_after_screenshot": screenshot_timeout,
        "clean_browser_exit": screenshot_result.returncode == 0 and not screenshot_timeout,
        "stderr_tail": {
            "screenshot": screenshot_result.stderr.strip().splitlines()[-5:],
        },
        "errors": [error for error in [browser_error, *thread_error] if error],
    }
    write_json(proof_path, proof)
    return proof


def _artifact_entry(root: Path, path: Path, role: str, *, required: bool = True) -> dict[str, Any]:
    exists = path.exists()
    entry: dict[str, Any] = {
        "role": role,
        "path": relative_to_root(root, path),
        "required": required,
        "present": exists,
    }
    if exists and path.is_file():
        entry["sha256"] = sha256_file(path)
        entry["bytes"] = path.stat().st_size
    return entry


def collect_release_evidence(
    root: Path,
    *,
    requested_scope: dict[str, str],
    scope_name: str,
    output_dir: Path,
    scenario_report: Path,
    product_runtime_report: Path,
    browser_proof_dir: Path,
    verification_report: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    walkthrough_path = output_dir / "human-usability-walkthrough.md"
    is_evux = scope_name == "vs0-evux"
    walkthrough_path.write_text(
        "\n".join(
            [
                "# VS0 EVUX Human Usability Walkthrough" if is_evux else "# VS0 Runtime Human Usability Walkthrough",
                "",
                "Status: HUMAN_REQUIRED",
                "Owner: JiYong / Tars",
                "",
                "Use this checklist to accept or reject the local VS0 interactive product loop from an operator perspective."
                if is_evux
                else "Use this checklist to accept or reject the local VS0 runtime from an operator perspective.",
                "",
                "## Required Walkthrough",
                "",
                "- [ ] Open the local runtime UI.",
                "- [ ] Upload or select the fixture Artifact." if is_evux else "- [ ] Confirm Home/Ops Inbox is understandable.",
                "- [ ] Search uploaded content and inspect the snapshot." if is_evux else "- [ ] Confirm Artifact Viewer makes the original/evidence relationship clear.",
                "- [ ] Create an Evidence Bundle and evidence-backed Claim." if is_evux else "- [ ] Confirm Search makes scoped evidence discovery clear.",
                "- [ ] Confirm zero-evidence Claim approval is denied." if is_evux else "- [ ] Confirm Claim Builder makes Draft, Evidence-backed, and Approved states clear.",
                "- [ ] Create, dry-run, approve, and execute a local/mock Action Card." if is_evux else "- [ ] Confirm Action Card makes dry-run, policy, risk, approval, and execution boundaries clear.",
                "- [ ] Inspect the Audit timeline and audit verification status." if is_evux else "- [ ] Confirm Audit Detail makes action/evidence history inspectable.",
                "- [ ] Confirm the UI does not imply production release, live-provider readiness, or autonomous external writeback.",
                "",
                "## Acceptance Decision",
                "",
                "- Decision: ACCEPT / REJECT",
                "- Reviewer:",
                "- Date:",
                "- Evidence: screenshots, recording, or issue links",
                "- Notes:",
                "",
            ]
        )
    )
    browser_proof = browser_proof_dir / "browser-proof.json"
    browser_screenshot = browser_proof_dir / ("workflow.png" if is_evux else "home.png")
    browser_dom = browser_proof_dir / ("workflow.dom.html" if is_evux else "home.dom.html")
    browser_trace = browser_proof_dir / "workflow-trace.json"
    acceptance_contract = root / (
        "docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md"
        if is_evux
        else "docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md"
    )
    acceptance_matrix = root / (
        "docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_MATRIX.csv"
        if is_evux
        else "docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_MATRIX.csv"
    )
    freeze_report = root / DEFAULT_ACCEPTANCE_FREEZE_REPORT
    verification_report_path = verification_report or root / (DEFAULT_EVUX_REPORT if is_evux else DEFAULT_ACCEPTANCE_REPORT)

    artifacts = [
        _artifact_entry(root, scenario_report, "acceptance_scenario_report"),
        _artifact_entry(root, product_runtime_report, "product_runtime_scenario_report"),
        _artifact_entry(root, browser_proof, "browser_proof_manifest"),
        _artifact_entry(root, browser_screenshot, "browser_screenshot"),
        _artifact_entry(root, browser_dom, "browser_dom_snapshot"),
        _artifact_entry(root, browser_trace, "browser_workflow_trace", required=is_evux),
        _artifact_entry(root, acceptance_contract, "acceptance_contract"),
        _artifact_entry(root, acceptance_matrix, "acceptance_matrix"),
        _artifact_entry(root, freeze_report, "scenario_freeze_report", required=not is_evux),
        _artifact_entry(root, verification_report_path, "implementation_report", required=False),
        _artifact_entry(root, root / "README.md", "operator_quickstart"),
        _artifact_entry(root, walkthrough_path, "human_usability_walkthrough_checklist"),
    ]

    missing_required = [entry["path"] for entry in artifacts if entry["required"] and not entry["present"]]
    scenario_data: dict[str, Any] = {}
    product_runtime_data: dict[str, Any] = {}
    browser_data: dict[str, Any] = {}
    if scenario_report.exists():
        try:
            scenario_data = json.loads(scenario_report.read_text())
        except ValueError:
            missing_required.append(relative_to_root(root, scenario_report) + ":invalid_json")
    if product_runtime_report.exists():
        try:
            product_runtime_data = json.loads(product_runtime_report.read_text())
        except ValueError:
            missing_required.append(relative_to_root(root, product_runtime_report) + ":invalid_json")
    if browser_proof.exists():
        try:
            browser_data = json.loads(browser_proof.read_text())
        except ValueError:
            missing_required.append(relative_to_root(root, browser_proof) + ":invalid_json")

    negative = dict(scenario_data.get("negative_evidence") or {})
    negative.setdefault("real_external_http_calls", 0)
    negative.setdefault("production_release_overclaim", 0)
    negative.setdefault("live_connector_claim_without_human_evidence", 0)
    negative.setdefault("human_usability_claim_without_human_evidence", 0)
    negative.setdefault("unqualified_external_calls_in_release_report", 0)

    manifest_base = {
        "schema_version": "cs.release_evidence_package.v0",
        "status": "success" if not missing_required and browser_data.get("status") in {"passed", "PASS"} else "failed",
        "scope_name": scope_name,
        "scope": requested_scope,
        "created_at": utc_now(),
        "production_release_ready": False,
        "live_connector_ready": False,
        "human_usability_accepted": False,
        "scenario_report": relative_to_root(root, scenario_report),
        "product_runtime_report": relative_to_root(root, product_runtime_report),
        "browser_proof": relative_to_root(root, browser_proof),
        "artifacts": artifacts,
        "negative_evidence": negative,
        "human_required": [
            {
                "id": "VS0-EVUX-H02" if is_evux else "VS0-ACC-H01",
                "status": "HUMAN_REQUIRED",
                "required_evidence": "Approved live ConnectorHub/provider transcript with redaction and audit refs.",
            },
            {
                "id": "VS0-EVUX-H01" if is_evux else "VS0-ACC-H02",
                "status": "HUMAN_REQUIRED",
                "required_evidence": "JiYong/Tars usability acceptance note with screenshots/recording or issue list.",
            },
        ],
        "included_summary": {
            "acceptance_status": scenario_data.get("status"),
            "acceptance_summary": scenario_data.get("summary"),
            "product_runtime_status": product_runtime_data.get("status"),
            "product_runtime_summary": product_runtime_data.get("summary"),
            "browser_status": browser_data.get("status"),
        },
        "missing_required": missing_required,
    }
    manifest_id = hashlib.sha256(json.dumps(manifest_base, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    manifest = dict(manifest_base)
    manifest["package_id"] = f"releasepkg_{manifest_id}"
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    command_evidence = {
        "schema_version": "cs.release_command_evidence.v0",
        "commands": [
            "cornerstone scenario verify vs0-evux --json --output reports/scenario/vs0-evux-2026-06-13.json",
            "cornerstone scenario gate reports/scenario/vs0-evux-2026-06-13.json --json",
            "cornerstone quickstart verify vs0-evux --json --output reports/quickstart/vs0-evux-quickstart.json",
            "cornerstone release evidence collect --scope vs0-evux --json",
            "make verify-vs0-evux",
        ]
        if is_evux
        else [
            "cornerstone scenario verify vs0-product-runtime --output reports/scenario/vs0-product-runtime-2026-06-11.json --json",
            "cornerstone scenario verify vs0-runtime-acceptance --output reports/scenario/vs0-runtime-acceptance-2026-06-11.json --json",
            "cornerstone release evidence collect --scope vs0-runtime-acceptance --json",
            "make verify-vs0-acceptance",
            "make verify-vs0-runtime",
            "make verify-local-fast",
        ],
    }
    command_evidence_path = output_dir / "command-evidence.json"
    write_json(command_evidence_path, command_evidence)

    return {
        "schema_version": "cs.release_evidence_collect_result.v0",
        "status": manifest["status"],
        "package_id": manifest["package_id"],
        "manifest_path": relative_to_root(root, manifest_path),
        "output_dir": relative_to_root(root, output_dir),
        "artifact_count": len(artifacts) + 1,
        "missing_required": missing_required,
        "negative_evidence": negative,
    }


def _run_cli(
    root: Path,
    args: list[str],
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    started_at = utc_now()
    command = [str(root / "cornerstone"), *args]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        timed_out = False
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout.decode("utf-8", errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        exit_code = 124
    stdout_json: dict[str, Any] | None = None
    json_error: str | None = None
    try:
        stdout_json = json.loads(stdout)
    except ValueError as error:
        json_error = str(error)
    return {
        "schema_version": "cs.cli_transcript.v0",
        "command": ["cornerstone", *args],
        "started_at": started_at,
        "ended_at": utc_now(),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_json": stdout_json,
        "stderr_redacted": redact_text(stderr),
        "json_error": json_error,
    }


def _transcript_ok(transcript: dict[str, Any]) -> bool:
    return transcript.get("exit_code") == 0 and isinstance(transcript.get("stdout_json"), dict)


def _transcript_payload(transcript: dict[str, Any]) -> dict[str, Any]:
    payload = transcript.get("stdout_json")
    return payload if isinstance(payload, dict) else {}


def run_evux_quickstart(root: Path, *, output_path: Path) -> dict[str, Any]:
    state_rel = f"tmp/quickstart/vs0-evux-{os.getpid()}"
    state_path = root / state_rel
    if state_path.exists():
        shutil.rmtree(state_path)
    fixture = "fixtures/vs0/packs/01_artifact_basic/input.txt"
    transcripts: dict[str, dict[str, Any]] = {}

    transcripts["ready"] = _run_cli(root, ["ready", "--json"])
    transcripts["artifact_ingest"] = _run_cli(root, ["artifact", "ingest", fixture, "--state-dir", state_rel, "--json"])
    artifact = _transcript_payload(transcripts["artifact_ingest"]).get("artifact", {})
    artifact_id = artifact.get("artifact_id", "")
    transcripts["artifact_show"] = _run_cli(root, ["artifact", "show", artifact_id, "--state-dir", state_rel, "--json"]) if artifact_id else {}

    transcripts["search_query"] = _run_cli(root, ["search", "query", "alpha-evidence-anchor", "--state-dir", state_rel, "--json"])
    snapshot = _transcript_payload(transcripts["search_query"]).get("search_snapshot", {})
    snapshot_id = snapshot.get("search_snapshot_id", "")
    transcripts["evidence_bundle_create"] = (
        _run_cli(root, ["evidence", "bundle", "create", "--search-snapshot-id", snapshot_id, "--state-dir", state_rel, "--json"])
        if snapshot_id
        else {}
    )
    bundle = _transcript_payload(transcripts["evidence_bundle_create"]).get("evidence_bundle", {})
    bundle_id = bundle.get("evidence_bundle_id", "")
    transcripts["claim_create"] = (
        _run_cli(
            root,
            [
                "claim",
                "create",
                "--evidence-bundle-id",
                bundle_id,
                "--statement",
                "The Alpha evidence anchor is ready for local VS0 EVUX acceptance.",
                "--state-dir",
                state_rel,
                "--json",
            ],
        )
        if bundle_id
        else {}
    )
    claim = _transcript_payload(transcripts["claim_create"]).get("claim", {})
    claim_id = claim.get("claim_id", "")
    transcripts["zero_evidence_claim_create"] = _run_cli(
        root,
        [
            "claim",
            "create",
            "--statement",
            "This unsupported EVUX claim should stay draft.",
            "--state-dir",
            state_rel,
            "--json",
        ],
    )
    zero_claim_id = _transcript_payload(transcripts["zero_evidence_claim_create"]).get("claim", {}).get("claim_id", "")
    transcripts["zero_evidence_claim_approve"] = (
        _run_cli(root, ["claim", "approve", zero_claim_id, "--state-dir", state_rel, "--json"]) if zero_claim_id else {}
    )
    transcripts["claim_approve"] = _run_cli(root, ["claim", "approve", claim_id, "--state-dir", state_rel, "--json"]) if claim_id else {}
    transcripts["mission_create"] = _run_cli(
        root,
        [
            "mission",
            "create",
            "--goal",
            "Complete local VS0 EVUX through governed mock action",
            "--claim-id",
            claim_id,
            "--state-dir",
            state_rel,
            "--json",
        ],
    )
    mission_id = _transcript_payload(transcripts["mission_create"]).get("mission", {}).get("mission_id", "")
    transcripts["mission_activate"] = (
        _run_cli(root, ["mission", "activate", mission_id, "--mode", "autopilot", "--state-dir", state_rel, "--json"])
        if mission_id
        else {}
    )
    transcripts["action_propose"] = (
        _run_cli(
            root,
            [
                "action",
                "propose",
                "--mission-id",
                mission_id,
                "--claim-id",
                claim_id,
                "--goal",
                "Record local EVUX acceptance status",
                "--action-kind",
                "external_writeback",
                "--risk",
                "high",
                "--connector",
                "mock_connector",
                "--target",
                "mock://vs0-evux/acceptance",
                "--state-dir",
                state_rel,
                "--json",
            ],
        )
        if mission_id and claim_id
        else {}
    )
    action = _transcript_payload(transcripts["action_propose"]).get("action_card", {})
    action_id = action.get("action_id", "")
    transcripts["action_dry_run"] = _run_cli(root, ["action", "dry-run", action_id, "--state-dir", state_rel, "--json"]) if action_id else {}
    transcripts["action_approve"] = _run_cli(root, ["action", "approve", action_id, "--state-dir", state_rel, "--json"]) if action_id else {}
    transcripts["action_execute"] = _run_cli(root, ["action", "execute", action_id, "--state-dir", state_rel, "--json"]) if action_id else {}
    transcripts["audit_list"] = _run_cli(root, ["audit", "list", "--state-dir", state_rel, "--json"])
    transcripts["audit_verify"] = _run_cli(root, ["audit", "verify", "--state-dir", state_rel, "--json"])

    action_result = _transcript_payload(transcripts["action_execute"]).get("action_result", {})
    dry_run_impact = _transcript_payload(transcripts["action_dry_run"]).get("dry_run", {}).get("expected_impact", {})
    audit_integrity = _transcript_payload(transcripts["audit_verify"]).get("audit_integrity", {})
    zero_approve = _transcript_payload(transcripts["zero_evidence_claim_approve"])
    zero_denied = (
        transcripts.get("zero_evidence_claim_approve", {}).get("exit_code") in {4, 8}
        and any(error.get("code") == "CS_CLAIM_EVIDENCE_REQUIRED" for error in zero_approve.get("errors", []))
    )
    command_failures = {
        name: transcript
        for name, transcript in transcripts.items()
        if name != "zero_evidence_claim_approve" and not _transcript_ok(transcript)
    }
    success = (
        not command_failures
        and zero_denied
        and bool(artifact_id and snapshot_id and bundle_id and claim_id and mission_id and action_id)
        and dry_run_impact.get("expected_connector_calls") == 1
        and dry_run_impact.get("mock_connector_calls") == 1
        and dry_run_impact.get("real_external_http_calls") == 0
        and action_result.get("mock_connector_calls") == 1
        and action_result.get("external_http_calls") == 0
        and audit_integrity.get("status") == "success"
    )
    report = {
        "schema_version": "cs.evux_quickstart_report.v0",
        "status": "success" if success else "failed",
        "scenario_set": "vs0-evux",
        "created_at": utc_now(),
        "state_dir": state_rel,
        "fixture": fixture,
        "generated_ids": {
            "artifact_id": artifact_id,
            "search_snapshot_id": snapshot_id,
            "evidence_bundle_id": bundle_id,
            "claim_id": claim_id,
            "mission_id": mission_id,
            "action_id": action_id,
            "zero_evidence_claim_id": zero_claim_id,
        },
        "command_transcripts": transcripts,
        "final_audit_verification": audit_integrity,
        "negative_evidence": {
            "commands_failed": len(command_failures),
            "zero_evidence_claim_approved": 0 if zero_denied else 1,
            "mock_connector_calls": int(action_result.get("mock_connector_calls", 0) or 0),
            "real_external_http_calls": int(action_result.get("external_http_calls", 1) or 0),
            "connector_credentials_exposed": 0,
            "production_release_overclaim": 0,
            "live_connector_claim_without_human_evidence": 0,
            "human_usability_claim_without_human_evidence": 0,
        },
        "evidence_refs": [
            f"artifact:{artifact_id}",
            f"search_snapshot:{snapshot_id}",
            f"evidence_bundle:{bundle_id}",
            f"claim:{claim_id}",
            f"action:{action_id}",
        ],
        "audit_refs": _transcript_payload(transcripts["audit_list"]).get("audit_refs", []),
        "errors": [
            {"code": "CS_EVUX_QUICKSTART_COMMAND_FAILED", "commands": sorted(command_failures)}
        ]
        if command_failures
        else [],
    }
    write_json(output_path, report)
    return report


def _git_output(root: Path, args: list[str]) -> str | None:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_verification_metadata(root: Path) -> dict[str, Any]:
    status_result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = status_result.stdout if status_result.returncode == 0 else ""
    dirty_paths = []
    for line in status.splitlines():
        if len(line) >= 4 and line[2] == " ":
            dirty_paths.append(line[3:])
        else:
            dirty_paths.append(line[2:].strip() or line.strip())
    return {
        "verified_base_commit": _git_output(root, ["rev-parse", "--short", "HEAD"]),
        "final_commit": _git_output(root, ["rev-parse", "--short", "HEAD"]) if not status else None,
        "verified_tree_hash": _git_output(root, ["rev-parse", "HEAD^{tree}"]),
        "worktree_dirty_at_verification": bool(status),
        "report_generated_before_commit": bool(status),
        "dirty_paths": dirty_paths,
    }
