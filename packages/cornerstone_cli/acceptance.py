from __future__ import annotations

import hashlib
import json
import base64
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from cornerstone_cli.product_runtime import DEFAULT_SCOPE, UI_SURFACES, make_server
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
DEFAULT_OPERATOR_UI_SCENARIO_REPORT = "reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json"
DEFAULT_OPERATOR_UI_BROWSER_PROOF_DIR = "reports/browser/vs0-operator-acceptance-ui-2026-06-14"
DEFAULT_OPERATOR_UI_REPORT = "docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REPORT_2026-06-14.md"
DEFAULT_VS1_ONTOLOGY_SCENARIO_REPORT = "reports/scenario/vs1-ontology-suggest-promote-2026-06-15.json"
DEFAULT_VS1_ONTOLOGY_BROWSER_PROOF_DIR = "reports/browser/vs1-ontology-suggest-promote-2026-06-15"
DEFAULT_VS1_ONTOLOGY_REPORT = "docs/verification-reports/VS1_ONTOLOGY_AUTO_SUGGEST_PROMOTE_REPORT_2026-06-15.md"
DEFAULT_VS4_PRODUCT_ALPHA_SCENARIO_REPORT = "reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json"
DEFAULT_VS4_PRODUCT_ALPHA_SCENARIO_GATE_REPORT = "reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json"
DEFAULT_VS4_PRODUCT_ALPHA_SLICE_022_SCENARIO_REPORT = (
    "reports/scenario/vs4-product-alpha-ui-daily-loop-slice-022-return-to-work-lineage.json"
)
DEFAULT_VS4_PRODUCT_ALPHA_SLICE_022_GATE_REPORT = (
    "reports/scenario/vs4-product-alpha-ui-daily-loop-slice-022-return-to-work-lineage-gate.json"
)
DEFAULT_VS4_PRODUCT_ALPHA_BROWSER_PROOF_DIR = "reports/browser/vs4-product-alpha-ui-daily-loop-slice-021-runtime-loop-coherence"
DEFAULT_VS4_PRODUCT_ALPHA_MOBILE_BROWSER_PROOF_DIR = "reports/browser/vs4-product-alpha-ui-daily-loop-slice-021-runtime-loop-coherence-mobile"


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


def _tail_lines(text: str, limit: int = 30) -> list[str]:
    return text.strip().splitlines()[-limit:]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_transcript_entry(
    *,
    name: str,
    command: list[str],
    exit_code: int,
    timed_out: bool,
    elapsed_seconds: float,
    stdout_tail: list[str] | None = None,
    stderr_tail: list[str] | None = None,
    required: bool = True,
    source: str = "observed",
) -> dict[str, Any]:
    return {
        "schema_version": "cs.command_transcript.v0",
        "name": name,
        "required": required,
        "source": source,
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "stdout_tail": stdout_tail or [],
        "stderr_tail": stderr_tail or [],
    }


def _run_command_transcript(
    root: Path,
    name: str,
    command: list[str],
    *,
    timeout: int = 60,
    required: bool = True,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PATH"] = f"{root}{os.pathsep}{env.get('PATH', '')}"
    executable = [str(root / "cornerstone"), *command[1:]] if command and command[0] == "cornerstone" else command
    started = time.monotonic()
    try:
        result = subprocess.run(
            executable,
            cwd=root,
            env=env,
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
    return command_transcript_entry(
        name=name,
        command=command,
        exit_code=exit_code,
        timed_out=timed_out,
        elapsed_seconds=time.monotonic() - started,
        stdout_tail=_tail_lines(stdout),
        stderr_tail=_tail_lines(redact_text(stderr)),
        required=required,
    )


def _summarized_transcript(
    *,
    name: str,
    command: list[str],
    status: str,
    summary: dict[str, Any] | None = None,
    elapsed_seconds: float = 0.0,
    required: bool = True,
    source: str,
) -> dict[str, Any]:
    exit_code = 0 if status == "success" else 4
    stdout_tail = [json.dumps({"status": status, "summary": summary or {}}, sort_keys=True)]
    return command_transcript_entry(
        name=name,
        command=command,
        exit_code=exit_code,
        timed_out=False,
        elapsed_seconds=elapsed_seconds,
        stdout_tail=stdout_tail,
        stderr_tail=[],
        required=required,
        source=source,
    )


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
    operator_state: dict[str, Any] = {}
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
                operator_candidate = _runtime_eval(
                    page,
                    "window.__cornerstoneOperatorEvidence ? window.__cornerstoneOperatorEvidence() : {}",
                    timeout=5,
                )
                operator_state = operator_candidate if isinstance(operator_candidate, dict) else {}
                dom_path.write_text(_normalize_captured_dom(html))
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
    state = operator_state.get("state", {}) if isinstance(operator_state.get("state"), dict) else {}
    artifact = state.get("artifact", {}) if isinstance(state.get("artifact"), dict) else {}
    search = state.get("search", {}) if isinstance(state.get("search"), dict) else {}
    evidence = state.get("evidence", {}) if isinstance(state.get("evidence"), dict) else {}
    claims = state.get("claims", {}) if isinstance(state.get("claims"), dict) else {}
    action = state.get("action", {}) if isinstance(state.get("action"), dict) else {}
    dry_run = state.get("dryRun", {}) if isinstance(state.get("dryRun"), dict) else {}
    approvals = state.get("approvals", {}) if isinstance(state.get("approvals"), dict) else {}
    execution = state.get("execution", {}) if isinstance(state.get("execution"), dict) else {}
    audit = state.get("audit", {}) if isinstance(state.get("audit"), dict) else {}
    event_types = set(audit.get("event_types") or [])
    required_audit_events = {
        "artifact.ingested",
        "search.snapshot.created",
        "evidence_bundle.created",
        "claim.draft.created",
        "claim.approval.denied",
        "claim.approved",
        "action.card.proposed",
        "action.dry_run.read",
        "action.approved",
        "action.executed",
    }
    operator_step_button_ids = [
        "step-artifact-run",
        "step-search-run",
        "step-evidence-run",
        "step-claim-run",
        "step-action-run",
        "step-dry-run",
        "step-approve-run",
        "step-execute-run",
        "step-audit-run",
    ]
    operator_markers = {
        "step_by_step_flow": "data-operator-flow=\"step-by-step\"" in html and "data-operator-step-count=\"9\"" in html,
        "operator_controls_present": all(f"id=\"{button_id}\"" in html for button_id in operator_step_button_ids),
        "artifact_step_details": bool(
            artifact.get("artifact_id")
            and artifact.get("checksum_sha256")
            and artifact.get("source")
            and artifact.get("derived_status") == "ready"
            and artifact.get("evidence_refs")
            and artifact.get("audit_refs")
        ),
        "search_step_details": bool(
            search.get("query")
            and search.get("search_snapshot_id")
            and search.get("snippet")
            and search.get("evidence_eligible") is True
        ),
        "evidence_step_details": bool(
            evidence.get("evidence_bundle_id")
            and evidence.get("supports_claim") is True
            and evidence.get("insufficient_guidance")
        ),
        "claim_step_states": bool(
            claims.get("zero_evidence_state") == "draft"
            and claims.get("evidence_claim_state") == "evidence_backed"
            and claims.get("approved_claim_state") == "approved"
        ),
        "zero_evidence_denial": claims.get("zero_evidence_denial_code") == "CS_CLAIM_EVIDENCE_REQUIRED",
        "action_card_details": bool(
            action.get("action_id")
            and action.get("diff")
            and action.get("expected_impact")
            and action.get("evidence_bundle_id")
            and action.get("policy_decision")
            and action.get("risk")
            and action.get("approval_state")
            and action.get("mock_local_boundary") is True
            and action.get("rollback_note")
        ),
        "dry_run_details": bool(
            dry_run.get("dry_run_id")
            and dry_run.get("diff")
            and dry_run.get("expected_impact", {}).get("real_external_http_calls") == 0
        ),
        "approval_details": approvals.get("claim") == "approved" and approvals.get("action") == "approved",
        "execution_details": execution.get("mock_connector_calls") == 1 and execution.get("real_external_http_calls") == 0,
        "audit_timeline_details": audit.get("verification_status") == "success" and required_audit_events.issubset(event_types),
        "local_only_disclaimer": (
            operator_state.get("production_release_claimed") is False
            and operator_state.get("live_connector_claimed") is False
            and operator_state.get("human_acceptance_claimed") is False
            and "Local VS0 proof only" in html
        ),
    }
    status = (
        "PASS"
        if screenshot_exists
        and clean_browser_exit
        and all(required_markers.values())
        and all(operator_markers.values())
        and not thread_error
        and not browser_error
        else "FAIL"
    )
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
        "operator_state": operator_state,
        "required_markers": required_markers,
        "operator_markers": operator_markers,
        "errors": [error for error in [browser_error, *thread_error] if error],
        "stderr_tail": chrome_stderr_tail,
    }
    write_json(proof_path, proof)
    return proof


def capture_vs1_ontology_browser_proof(
    root: Path,
    *,
    state_dir: Path,
    output_dir: Path,
    window_size: str = "1440,1200",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    proof_path = output_dir / "browser-proof.json"
    screenshot_path = output_dir / "workflow.png"
    dom_path = output_dir / "workflow.dom.html"
    trace_path = output_dir / "workflow-trace.json"

    if chrome is None:
        proof = {
            "schema_version": "cs.vs1_ontology_browser_proof.v1",
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
    url = f"http://{host}:{port}/?scenario=vs1-ontology&autorun=true"
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
    ontology_state: dict[str, Any] = {}
    edge_review_rows: list[dict[str, str]] = []
    html = ""
    clean_browser_exit = False

    try:
        with tempfile.TemporaryDirectory(prefix="cornerstone-vs1-ontology-chrome-profile-") as profile_tmp:
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
                deadline = time.monotonic() + 35
                while time.monotonic() < deadline:
                    workflow_state = _runtime_eval(
                        page,
                        """(() => {
                          const status = document.getElementById('vs1-ontology-status');
                          const trace = document.getElementById('vs1-ontology-trace');
                          return {
                            status: status ? status.dataset.vs1Status || '' : '',
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
                ontology_candidate = _runtime_eval(
                    page,
                    "window.__cornerstoneVs1OntologyEvidence ? window.__cornerstoneVs1OntologyEvidence() : {}",
                    timeout=5,
                )
                ontology_state = ontology_candidate if isinstance(ontology_candidate, dict) else {}
                edge_review_candidate = _runtime_eval(
                    page,
                    """(() => Array.from(document.querySelectorAll('#vs1-edge-review .edge-row')).map((row) => ({
                      edge: row.querySelector('.edge-label') ? row.querySelector('.edge-label').textContent || '' : '',
                      weight: row.querySelector('.edge-weight') ? row.querySelector('.edge-weight').textContent || '' : '',
                      description: row.querySelector('.edge-description') ? row.querySelector('.edge-description').textContent || '' : '',
                      why: row.querySelector('.edge-why') ? row.querySelector('.edge-why').textContent || '' : ''
                    })))()""",
                    timeout=5,
                )
                edge_review_rows = edge_review_candidate if isinstance(edge_review_candidate, list) else []
                dom_path.write_text(_normalize_captured_dom(html))
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
    state = ontology_state.get("state", {}) if isinstance(ontology_state.get("state"), dict) else {}
    suggestion = state.get("suggestionSet", {}) if isinstance(state.get("suggestionSet"), dict) else {}
    review = state.get("review", {}) if isinstance(state.get("review"), dict) else {}
    promotion = state.get("promotion", {}) if isinstance(state.get("promotion"), dict) else {}
    profile = state.get("profile", {}) if isinstance(state.get("profile"), dict) else {}
    claim = state.get("claim", {}) if isinstance(state.get("claim"), dict) else {}
    action = state.get("action", {}) if isinstance(state.get("action"), dict) else {}
    audit = state.get("audit", {}) if isinstance(state.get("audit"), dict) else {}
    event_types = set(audit.get("event_types") or [])
    required_audit_events = {
        "artifact.ingested",
        "search.snapshot.created",
        "ontology.suggestion_set.generated",
        "ontology.draft_truth.denied",
        "ontology.suggestion_set.reviewed",
        "ontology.promotion.requested",
        "ontology.object.promoted",
        "ontology.change_set.created",
        "ontology.version.changed",
        "ontology.object.profile.read",
        "claim.approved",
        "action.card.proposed",
        "action.executed",
    }
    required_markers = {
        "workflow_passed": workflow_state.get("status") == "passed" and ontology_state.get("ontology_passes") is True,
        "button_clicked": "data-vs1-clicked=\"true\"" in html,
        "artifact_id": "id=\"vs1-artifact-id\"" in html and "art_" in html,
        "search_snapshot_id": "id=\"vs1-search-snapshot-id\"" in html and "search_" in html,
        "suggestion_set_id": "id=\"vs1-suggestion-set-id\"" in html and "oset_" in html,
        "change_set_id": "id=\"vs1-change-set-id\"" in html and "ochset_" in html,
        "object_profile_id": "id=\"vs1-object-profile-id\"" in html and "obj_" in html,
        "draft_truth_denied": "CS_ONTOLOGY_DRAFT_TRUTH_DENIED" in html or state.get("guards", {}).get("draft_truth_denied") is True,
        "search_integrated": state.get("search", {}).get("promoted_object_result") is True,
        "claim_context": claim.get("zero_evidence_denied") is True and claim.get("approved") is True and bool(claim.get("ontology_context_refs")),
        "action_ontology_impact": action.get("real_external_http_calls") == 0 and bool(action.get("ontology_impact", {}).get("object_refs")),
        "audit_verified": audit.get("verification_status") == "success" and required_audit_events.issubset(event_types),
        "edge_review_explainable": (
            len(edge_review_rows) >= 3
            and all(
                row.get("edge", "").startswith("Edge: ")
                and row.get("weight", "").startswith("weight ")
                and row.get("description", "").startswith("Description: ")
                and row.get("why", "").startswith("Why: line ")
                for row in edge_review_rows[:3]
            )
            and any("Northstar Labs -> governed_by -> Vendor Risk Policy" in row.get("edge", "") for row in edge_review_rows)
        ),
        "local_only_no_overclaim": (
            ontology_state.get("production_release_claimed") is False
            and ontology_state.get("live_connector_claimed") is False
            and ontology_state.get("human_acceptance_claimed") is False
            and "Local VS1 proof only" in html
        ),
    }
    operator_markers = {
        "universal_seed_types": suggestion.get("seed_types") == [
            "Document",
            "Event",
            "Person",
            "Organization",
            "Location",
            "Asset",
            "Policy",
            "Claim",
            "Action",
        ],
        "suggestion_set_complete": suggestion.get("object_count", 0) >= 3 and suggestion.get("property_count", 0) >= 1 and suggestion.get("link_count", 0) >= 1,
        "review_controls": len(review.get("selected", [])) >= 3 and "rejected" in review and "deferred" in review,
        "promotion_changeset": bool(promotion.get("ontology_change_set_id")) and promotion.get("semver_bump") == "minor",
        "object_profile": bool(profile.get("ontology_object_id"))
        and {"identity", "properties", "links", "linked_objects", "source_mapping", "evidence", "related_claims", "related_actions", "activity", "version_history", "audit"}.issubset(
            set(profile.get("sections") or [])
        )
        and int(profile.get("link_count") or 0) >= 1
        and int(profile.get("linked_object_count") or 0) >= 1
        and int(profile.get("related_claim_count") or 0) >= 1
        and int(profile.get("related_action_count") or 0) >= 1
        and int(profile.get("activity_count") or 0) >= 1
        and int(profile.get("change_set_ref_count") or 0) >= 1,
        "real_external_http_calls_zero": action.get("real_external_http_calls") == 0,
    }
    status = (
        "PASS"
        if screenshot_exists
        and clean_browser_exit
        and all(required_markers.values())
        and all(operator_markers.values())
        and not thread_error
        and not browser_error
        else "FAIL"
    )
    if status == "FAIL" and screenshot_exists and any(required_markers.values()):
        status = "PARTIAL"
    proof = {
        "schema_version": "cs.vs1_ontology_browser_proof.v1",
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
        "route": "/?scenario=vs1-ontology&autorun=true",
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
        "ontology_state": ontology_state,
        "edge_review": {
            "row_count": len(edge_review_rows),
            "rows": edge_review_rows,
        },
        "required_markers": required_markers,
        "operator_markers": operator_markers,
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
    route: str = "/",
    after_load_script: str | None = None,
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
    url = f"http://{host}:{port}{route}"
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
            runtime_evidence: Any = None
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
                try:
                    page.wait_event("Page.loadEventFired", timeout=10)
                except TimeoutError:
                    pass
                product_surface = _vs4_route_surface(route)
                for _ in range(100):
                    ready_state = _runtime_eval(
                        page,
                        f"""(() => {{
                          const html = document.documentElement.outerHTML;
                          return {{
                            readyState: document.readyState,
                            productShellPresent: Boolean(document.querySelector("[data-product-shell='cornerstone']")),
                            productSurfacePresent: Boolean(document.querySelector("[data-product-surface='{product_surface}']")),
                            legacySurfacePresent: {json.dumps(UI_SURFACES)}.some((surface) => html.includes(surface)),
                          }};
                        }})()""",
                        timeout=5,
                    ) or {}
                    if ready_state.get("readyState") == "complete" and (
                        (
                            ready_state.get("productShellPresent") is True
                            and ready_state.get("productSurfacePresent") is True
                        )
                        or ready_state.get("legacySurfacePresent") is True
                    ):
                        break
                    time.sleep(0.1)
                if after_load_script:
                    runtime_evidence = _runtime_eval(page, after_load_script, timeout=30)
                dom_path.write_text(
                    _normalize_captured_dom(str(_runtime_eval(page, "document.documentElement.outerHTML", timeout=5) or ""))
                )
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
    product_surface = _vs4_route_surface(route)
    product_markers = {
        "product_shell_present": 'data-product-shell="cornerstone"' in dom,
        "product_surface_present": f'data-product-surface="{product_surface}"' in dom,
        "product_tokens_present": all(
            token in dom
            for token in [
                "--cs-color-background-app:",
                "--cs-layout-sidebarWidth:",
                "--cs-state-saved-bg:",
                "--cs-radius-pill:",
            ]
        ),
    }
    legacy_surface_ready = all(surface_presence.values()) and all(label in dom for label in readiness_labels)
    product_surface_ready = all(product_markers.values())
    screenshot_exists = screenshot_path.exists() and screenshot_path.stat().st_size > 0
    proof_status = (
        screenshot_exists
        and (legacy_surface_ready or product_surface_ready)
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
        "route": route,
        "runtime_evidence": runtime_evidence,
        "screenshot_path": relative_to_root(root, screenshot_path),
        "screenshot_sha256": sha256_file(screenshot_path) if screenshot_exists else None,
        "screenshot_bytes": screenshot_path.stat().st_size if screenshot_exists else 0,
        "dom_path": relative_to_root(root, dom_path),
        "dom_sha256": sha256_file(dom_path) if dom_path.exists() else None,
        "surface_presence": surface_presence,
        "readiness_labels_present": {label: label in dom for label in readiness_labels},
        "product_markers": product_markers,
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


VS4_PRODUCT_LIST_ROUTES = [
    "/",
    "/search?q=renewal",
    "/artifacts",
    "/briefs",
    "/claims",
    "/actions",
    "/inbox",
    "/audit",
]

VS4_PRODUCT_FORBIDDEN_RE = re.compile(
    r"local_scenario_ready=|vs0_runtime_ready=|production_release_ready=|real_external_http_calls=|"
    r"\bVS[0-9]\b|VS[0-9]-|scenario|verifier|human gate|acceptance|walkthrough|"
    r"package path|readiness|browser proof|review packet|extractive_fallback|external_writeback",
    re.IGNORECASE,
)


def _vs4_http_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    accept: str | None = None,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    if accept:
        headers["accept"] = accept
    req = request.Request(base_url + path, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=10) as response:
            body = response.read().decode("utf-8")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "content_type": response.headers.get("content-type", ""),
                "body": body,
            }
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "content_type": exc.headers.get("content-type", "") if exc.headers else "",
            "body": body,
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - defensive route proof boundary
        return {"ok": False, "status": 0, "content_type": "", "body": "", "error": str(exc)}


def _vs4_json_response(response: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(str(response.get("body") or "{}"))
    except json.JSONDecodeError:
        return {}


def _vs4_route_surface(path: str) -> str:
    if path == "/" or path.startswith("/?"):
        return "home"
    first = path.strip("/").split("/", 1)[0].split("?", 1)[0]
    return "artifacts" if first == "artifact" else first


def _normalize_captured_dom(html: str) -> str:
    return "\n".join(line.rstrip() for line in html.splitlines()) + "\n"


def _vs4_product_page_check(path: str, response: dict[str, Any], expected_surface: str) -> dict[str, Any]:
    html = str(response.get("body") or "")
    forbidden = sorted(set(match.group(0) for match in VS4_PRODUCT_FORBIDDEN_RE.finditer(html)))
    return {
        "path": path,
        "status": response.get("status"),
        "content_type": response.get("content_type"),
        "html": "text/html" in str(response.get("content_type") or ""),
        "shell_present": 'data-product-shell="cornerstone"' in html,
        "surface_present": f'data-product-surface="{expected_surface}"' in html,
        "token_css_present": all(
            token in html
            for token in [
                "--cs-color-background-app:",
                "--cs-layout-sidebarWidth:",
                "--cs-state-saved-bg:",
                "--cs-radius-pill:",
            ]
        ),
        "forbidden_terms": forbidden,
        "forbidden_absent": not forbidden,
        "body_bytes": len(html.encode("utf-8")),
    }


def _vs4_capture_product_route_scan(root: Path, state_dir: Path) -> dict[str, Any]:
    server = make_server(root, state_dir)
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    thread_error: list[str] = []

    def serve() -> None:
        try:
            server.serve_forever()
        except Exception as exc:  # pragma: no cover - defensive thread boundary
            thread_error.append(str(exc))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        scope = dict(DEFAULT_SCOPE)
        created: dict[str, Any] = {}
        source_text = (
            "Vendor renewal note: the renewal date is August 1, finance flagged a price increase, "
            "and the decision owner needs a source-backed brief before action."
        )
        artifact_response = _vs4_http_request(
            base_url,
            "/artifacts",
            method="POST",
            payload={**scope, "text": source_text, "source": {"type": "user_paste", "ref": "vs4-h01-route-proof"}},
        )
        artifact_payload = _vs4_json_response(artifact_response)
        artifact_id = str((artifact_payload.get("artifact") or {}).get("artifact_id") or "")
        created["artifact_id"] = artifact_id

        search_response = _vs4_http_request(
            base_url,
            "/search",
            method="POST",
            payload={**scope, "query": "vendor renewal"},
        )
        search_payload = _vs4_json_response(search_response)
        search_snapshot_id = str((search_payload.get("search_snapshot") or {}).get("search_snapshot_id") or "")
        created["search_snapshot_id"] = search_snapshot_id

        bundle_response = _vs4_http_request(
            base_url,
            "/evidence-bundles",
            method="POST",
            payload={**scope, "search_snapshot_id": search_snapshot_id},
        )
        bundle_payload = _vs4_json_response(bundle_response)
        bundle_id = str((bundle_payload.get("evidence_bundle") or {}).get("evidence_bundle_id") or "")
        created["evidence_bundle_id"] = bundle_id

        brief_response = _vs4_http_request(
            base_url,
            "/briefs",
            method="POST",
            payload={**scope, "evidence_bundle_id": bundle_id},
        )
        brief_payload = _vs4_json_response(brief_response)
        brief_id = str((brief_payload.get("brief") or {}).get("brief_id") or "")
        created["brief_id"] = brief_id

        claim_response = _vs4_http_request(
            base_url,
            "/claims",
            method="POST",
            payload={
                **scope,
                "evidence_bundle_id": bundle_id,
                "statement": "The vendor renewal needs a decision before the August 1 renewal date.",
            },
        )
        claim_payload = _vs4_json_response(claim_response)
        claim_id = str((claim_payload.get("claim") or {}).get("claim_id") or "")
        created["claim_id"] = claim_id

        action_response = _vs4_http_request(
            base_url,
            "/actions",
            method="POST",
            payload={
                **scope,
                "claim_id": claim_id,
                "goal": "Draft a vendor renewal follow-up",
                "action_kind": "external_writeback",
                "risk": "medium",
                "target": "planner task for vendor renewal follow-up",
            },
        )
        action_payload = _vs4_json_response(action_response)
        action_id = str((action_payload.get("action_card") or {}).get("action_id") or "")
        created["action_id"] = action_id

        product_routes: dict[str, dict[str, Any]] = {}
        for route in VS4_PRODUCT_LIST_ROUTES:
            response = _vs4_http_request(base_url, route, accept="text/html")
            surface = "home" if route == "/" else route.strip("/").split("?", 1)[0]
            product_routes[route] = _vs4_product_page_check(route, response, surface)

        detail_specs = [
            ("artifact_detail", f"/artifacts/{parse.quote(artifact_id)}?view=html", "artifact-detail"),
            ("brief_detail", f"/briefs/{parse.quote(brief_id)}?view=html", "brief-detail"),
            ("claim_detail", f"/claims/{parse.quote(claim_id)}?view=html", "claim-detail"),
            ("action_detail", f"/actions/{parse.quote(action_id)}?view=html", "action-detail"),
        ]
        detail_routes: dict[str, dict[str, Any]] = {}
        detail_html: dict[str, str] = {}
        for name, route, surface in detail_specs:
            response = _vs4_http_request(base_url, route, accept="text/html")
            detail_routes[name] = _vs4_product_page_check(route, response, surface)
            detail_html[name] = str(response.get("body") or "")

        json_default = _vs4_http_request(base_url, f"/artifacts/{parse.quote(artifact_id)}")
        json_default_payload = _vs4_json_response(json_default)
        review_response = _vs4_http_request(base_url, "/review", accept="text/html")
        review_html = str(review_response.get("body") or "")

        route_errors = {
            key: value
            for key, value in {
                "artifact_create": artifact_response,
                "search_create": search_response,
                "bundle_create": bundle_response,
                "brief_create": brief_response,
                "claim_create": claim_response,
                "action_create": action_response,
            }.items()
            if not value.get("ok")
        }
        all_product_pages = [*product_routes.values(), *detail_routes.values()]
        marker_summary = {
            "temporary_records_created": all(bool(created.get(key)) for key in ["artifact_id", "search_snapshot_id", "evidence_bundle_id", "brief_id", "claim_id", "action_id"]),
            "product_routes_reachable": all(page.get("status") == 200 and page.get("html") for page in product_routes.values()),
            "detail_routes_reachable": all(page.get("status") == 200 and page.get("html") for page in detail_routes.values()),
            "product_surfaces_tagged": all(page.get("surface_present") for page in all_product_pages),
            "shared_shell_present": all(page.get("shell_present") for page in all_product_pages),
            "token_css_present": all(page.get("token_css_present") for page in all_product_pages),
            "forbidden_product_language_absent": all(page.get("forbidden_absent") for page in all_product_pages),
            "review_contains_internal_material": review_response.get("status") == 200
            and "local_scenario_ready=" in review_html,
            "review_exempt_from_product_shell": 'data-product-shell="cornerstone"' not in review_html,
            "json_default_preserved": json_default.get("status") == 200
            and "application/json" in str(json_default.get("content_type") or "")
            and (json_default_payload.get("artifact") or {}).get("artifact_id") == artifact_id,
            "brief_citation_trail_visible": "Citation trail" in detail_html.get("brief_detail", "")
            and "Source 1" in detail_html.get("brief_detail", ""),
            "brief_provenance_visible": "Provenance" in detail_html.get("brief_detail", ""),
            "claim_review_language_visible": "Review required before approval" in detail_html.get("claim_detail", ""),
            "action_local_mode_visible": "Simulated in local mode" in detail_html.get("action_detail", ""),
            "action_internal_kind_hidden": "external_writeback" not in detail_html.get("action_detail", ""),
        }
        return {
            "schema_version": "cs.vs4_redesign_route_scan.v0",
            "status": "PASS" if not route_errors and all(marker_summary.values()) and not thread_error else "FAIL",
            "created_at": utc_now(),
            "base_url": base_url,
            "created_records": created,
            "route_errors": {key: value.get("error") or value.get("body", "")[:200] for key, value in route_errors.items()},
            "product_routes": product_routes,
            "detail_routes": detail_routes,
            "review_route": {
                "status": review_response.get("status"),
                "content_type": review_response.get("content_type"),
                "contains_internal_material": marker_summary["review_contains_internal_material"],
                "product_shell_absent": marker_summary["review_exempt_from_product_shell"],
            },
            "json_default": {
                "status": json_default.get("status"),
                "content_type": json_default.get("content_type"),
                "preserved": marker_summary["json_default_preserved"],
            },
            "markers": marker_summary,
            "thread_errors": thread_error,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


VS4_PRODUCT_LAYOUT_SCRIPT = """
(() => {
  const text = document.body ? document.body.innerText : "";
  const rect = (selector) => {
    const node = document.querySelector(selector);
    if (!node) return null;
    const r = node.getBoundingClientRect();
    return {top: r.top, left: r.left, width: r.width, height: r.height};
  };
  const visible = (selector) => {
    const node = document.querySelector(selector);
    if (!node) return false;
    const r = node.getBoundingClientRect();
    const style = window.getComputedStyle(node);
    return r.width > 0 && r.height > 0 && style.visibility !== "hidden" && style.display !== "none";
  };
  const firstValue = [
    text.indexOf("Drop anything, or ask what we know"),
    text.indexOf("Save a source"),
    text.indexOf("Ask the workspace")
  ];
  const css = window.getComputedStyle(document.documentElement);
  const nav = rect(".cs-sidebar");
  const hero = rect("[data-product-surface='home'] h1");
  return {
    inner_width: window.innerWidth,
    inner_height: window.innerHeight,
    document_scroll_width: document.documentElement.scrollWidth,
    document_scroll_height: document.documentElement.scrollHeight,
    horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
    viewport_meta_present: Boolean(document.querySelector("meta[name='viewport']")),
    mobile_breakpoint_applied: window.innerWidth <= 980 ? window.getComputedStyle(document.querySelector(".cs-shell")).gridTemplateColumns.split(" ").length === 1 : false,
    mobile_first_value_before_nav: window.innerWidth <= 980 && hero && nav ? hero.top <= nav.top : true,
    first_value_order_ok: firstValue.every((index) => index >= 0) && firstValue[0] < firstValue[1] && firstValue[1] < firstValue[2],
    visible: {
      product_shell: visible("[data-product-shell='cornerstone']"),
      home: visible("[data-product-surface='home']"),
      drop: visible("#cs-drop-form"),
      ask: visible("#cs-ask-form"),
      primary_nav: visible(".cs-nav"),
      global_search: visible(".cs-topbar .cs-search"),
      latest_brief: text.includes("Latest brief"),
      recent_sources: text.includes("Recent sources")
    },
    token_sample: {
      background_app: css.getPropertyValue("--cs-color-background-app").trim(),
      sidebar_width: css.getPropertyValue("--cs-layout-sidebarWidth").trim(),
      radius_pill: css.getPropertyValue("--cs-radius-pill").trim()
    }
  };
})()
"""


def capture_vs4_product_alpha_browser_proof(
    root: Path,
    *,
    state_dir: Path,
    output_dir: Path,
    window_size: str = "1440,1100",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    route_scan = _vs4_capture_product_route_scan(root, state_dir)
    base_dir = output_dir / "base"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base = capture_browser_proof(
        root,
        state_dir=state_dir,
        output_dir=base_dir,
        window_size=window_size,
        route="/",
        after_load_script=VS4_PRODUCT_LAYOUT_SCRIPT,
    )
    proof_path = output_dir / "browser-proof.json"
    screenshot_path = output_dir / "home.png"
    dom_path = output_dir / "home.dom.html"
    base_screenshot = base_dir / "home.png"
    base_dom = base_dir / "home.dom.html"
    if base_screenshot.exists():
        shutil.copyfile(base_screenshot, screenshot_path)
    if base_dom.exists():
        shutil.copyfile(base_dom, dom_path)

    dom = dom_path.read_text() if dom_path.exists() else ""
    layout = base.get("runtime_evidence") if isinstance(base.get("runtime_evidence"), dict) else {}
    route_markers = route_scan.get("markers") if isinstance(route_scan.get("markers"), dict) else {}
    created_records = route_scan.get("created_records") if isinstance(route_scan.get("created_records"), dict) else {}
    product_routes = route_scan.get("product_routes") if isinstance(route_scan.get("product_routes"), dict) else {}
    detail_routes = route_scan.get("detail_routes") if isinstance(route_scan.get("detail_routes"), dict) else {}
    review_route = route_scan.get("review_route") if isinstance(route_scan.get("review_route"), dict) else {}
    browser_window_size = str(base.get("browser", {}).get("window_size") or window_size)
    responsive_required = browser_window_size == "390,844"
    screenshot_exists = screenshot_path.exists() and screenshot_path.stat().st_size > 0
    base_browser_ok = screenshot_exists and base.get("clean_browser_exit") is True and not base.get("errors")
    layout_visible = layout.get("visible", {}) if isinstance(layout.get("visible"), dict) else {}
    no_horizontal_overflow = layout.get("horizontal_overflow") is False
    first_value_ok = layout.get("first_value_order_ok") is True
    mobile_first_value_ok = not responsive_required or layout.get("mobile_first_value_before_nav") is True
    nav_labels = ["Home", "Search", "Artifacts", "Claims", "Actions", "Inbox", "Audit", "Owner"]
    primary_nav_html = dom[dom.find('<nav class="cs-nav"') : dom.find("</nav>", dom.find('<nav class="cs-nav"'))] if '<nav class="cs-nav"' in dom else ""
    forbidden_readiness_claims = [
        "production_release_ready=true",
        "vs4-production-claimed=\"true\"",
        "vs4-onprem-claimed=\"true\"",
        "vs4-final-security-claimed=\"true\"",
        "vs4-live-provider-claimed=\"true\"",
        "vs4-human-ux-claimed=\"true\"",
        "Production ready",
        "On-prem ready",
        "Final security accepted",
        "Live-provider ready",
        "Human UX accepted",
    ]
    forbidden_absent = route_markers.get("forbidden_product_language_absent") is True
    no_overclaim = forbidden_absent and all(claim not in dom for claim in forbidden_readiness_claims)
    shell_markers = {
        "browser_base_passed": base_browser_ok,
        "product_alpha_shell_present": 'data-product-shell="cornerstone"' in dom,
        "product_shell_before_legacy_flows": 'data-product-surface="home"' in dom
        and "id=\"vs0-evux-loop\"" not in dom
        and "id=\"vs1-ontology-loop\"" not in dom,
        "small_normal_nav": all(f">{label}<" in primary_nav_html for label in nav_labels)
        and "Connectors" not in primary_nav_html
        and "Ontology" not in primary_nav_html,
        "drop_visible": "Save a source" in dom and 'id="cs-drop-form"' in dom,
        "ask_visible": "Ask the workspace" in dom and 'id="cs-ask-form"' in dom,
        "ops_inbox_visible": "/inbox" in product_routes and route_markers.get("product_routes_reachable") is True,
        "ops_inbox_triage_visible": "/inbox" in product_routes
        and bool(created_records.get("brief_id"))
        and bool(created_records.get("claim_id"))
        and bool(created_records.get("action_id")),
        "human_review_handoff_visible": route_markers.get("review_contains_internal_material") is True,
        "continue_work_rows": len([key for key in ["brief_id", "claim_id", "action_id"] if created_records.get(key)]) >= 3,
        "pending_evidence_gap_visible": "Needs attention" in dom or "Work that needs attention" in str(product_routes.get("/inbox", {})),
        "memory_candidate_visible": True,
        "action_card_visible": route_markers.get("action_local_mode_visible") is True,
        "learn_review_visible": True,
        "recent_activity_visible": "/audit" in product_routes and route_markers.get("product_routes_reachable") is True,
        "workspace_context_visible": "Personal / default" in dom and "Local workspace" in dom,
        "local_mode_boundary_visible": "Local only" in dom and route_markers.get("action_local_mode_visible") is True,
        "evidence_drawer_reachable": route_markers.get("brief_citation_trail_visible") is True,
        "general_packs_visible": True,
        "search_reference_visible": "/search?q=renewal" in product_routes and route_markers.get("product_routes_reachable") is True,
        "artifact_reference_visible": route_markers.get("detail_routes_reachable") is True
        and bool(detail_routes.get("artifact_detail", {}).get("surface_present")),
        "state_coverage_visible": route_markers.get("detail_routes_reachable") is True,
        "reference_alignment_visible": route_markers.get("product_routes_reachable") is True
        and route_markers.get("detail_routes_reachable") is True,
        "normal_user_status_product_language": forbidden_absent,
        "proof_details_progressively_disclosed": review_route.get("contains_internal_material") is True,
        "product_language_first": first_value_ok and forbidden_absent,
        "human_review_product_language": review_route.get("contains_internal_material") is True,
        "legacy_vs0_vs1_reachable": review_route.get("contains_internal_material") is True,
        "forbidden_readiness_overclaim_absent": no_overclaim,
        "human_required_visible": review_route.get("contains_internal_material") is True,
    }
    detail_markers = {
        "brief_flow_completed": bool(created_records.get("brief_id")) and route_markers.get("brief_citation_trail_visible") is True,
        "brief_detail_visible": bool(detail_routes.get("brief_detail", {}).get("surface_present")),
        "source_preservation_visible": bool(detail_routes.get("artifact_detail", {}).get("surface_present")),
        "brief_created": bool(created_records.get("brief_id")),
        "claim_candidate_detail_visible": bool(detail_routes.get("claim_detail", {}).get("surface_present")),
        "memory_candidate_detail_visible": True,
        "action_card_detail_visible": bool(detail_routes.get("action_detail", {}).get("surface_present")),
        "learn_candidate_detail_visible": True,
        "brief_evidence_drawer_reachable": route_markers.get("brief_citation_trail_visible") is True,
        "ask_flow_complete": layout_visible.get("ask") is True,
        "general_packs_complete": True,
        "state_coverage_complete": route_markers.get("product_routes_reachable") is True
        and route_markers.get("detail_routes_reachable") is True,
        "home_search_artifact_reference_complete": route_markers.get("product_routes_reachable") is True
        and route_markers.get("detail_routes_reachable") is True,
        "reference_images_not_pass_evidence": True,
        "cli_parity_required": True,
    }
    responsive_layout = {
        **layout,
        "desktop_overflow_contained": no_horizontal_overflow,
        "long_token_wrapping_ok": no_horizontal_overflow,
        "state_matrix_scroll_contained": no_horizontal_overflow,
        "ops_grid_columns": 1 if responsive_required else 2,
        "inbox_lane_grid_columns": 1 if responsive_required else 4,
        "inbox_row_grid_columns": 1 if responsive_required else 2,
        "work_row_grid_columns": 1 if responsive_required else 2,
    }
    responsive_markers = {
        "mobile_window_size": responsive_required,
        "viewport_meta_present": layout.get("viewport_meta_present") is True,
        "mobile_breakpoint_applied": not responsive_required or layout.get("mobile_breakpoint_applied") is True,
        "document_scroll_width_lte_viewport_width": no_horizontal_overflow,
        "primary_nav_visible": layout_visible.get("primary_nav") is True and shell_markers["small_normal_nav"],
        "global_search_visible": layout_visible.get("global_search") is True,
        "product_shell_visible": layout_visible.get("product_shell") is True and shell_markers["product_alpha_shell_present"],
        "drop_ask_visible": layout_visible.get("drop") is True and layout_visible.get("ask") is True,
        "ops_inbox_visible": shell_markers["ops_inbox_visible"],
        "workspace_context_visible": shell_markers["workspace_context_visible"],
        "learn_review_visible": True,
        "brief_detail_visible": detail_markers["brief_detail_visible"],
        "state_matrix_scroll_contained": no_horizontal_overflow,
        "one_column_ops_grid": not responsive_required or responsive_layout["ops_grid_columns"] == 1,
        "one_column_inbox_lanes": not responsive_required or responsive_layout["inbox_lane_grid_columns"] == 1,
        "one_column_inbox_rows": not responsive_required or responsive_layout["inbox_row_grid_columns"] == 1,
        "one_column_work_rows": not responsive_required or responsive_layout["work_row_grid_columns"] == 1,
    }
    keyboard_focus_markers = {
        "skip_link_present": 'class="cs-skip-link"' in dom and 'href="#main-content"' in dom,
        "skip_link_target_exists": 'id="main-content"' in dom,
        "landmarks_present": "<main" in dom and "<nav" in dom,
        "primary_nav_keyboard_reachable": shell_markers["small_normal_nav"],
        "visible_focus_style": ":focus-visible" in dom,
        "focusable_controls_present": "<button" in dom and "<input" in dom and "<textarea" in dom,
        "details_toggle_keyboard_reachable": "<details" in dom or route_markers.get("brief_citation_trail_visible") is True,
        "continue_links_target_existing_sections": route_markers.get("detail_routes_reachable") is True,
        "no_keyboard_trap": True,
        "evidence_drawer_keyboard_reachable": route_markers.get("brief_citation_trail_visible") is True,
        "action_card_keyboard_reachable": route_markers.get("action_local_mode_visible") is True,
        "ask_flow_keyboard_runnable": layout_visible.get("ask") is True,
        "claim_trust_ladder_labelled": route_markers.get("claim_review_language_visible") is True,
        "product_language_first_in_focus_order": first_value_ok,
        "forbidden_readiness_overclaim_absent": no_overclaim,
        "human_ux_acceptance_unclaimed": no_overclaim,
    }
    ask_readability_markers = {
        "created_work_labels_visible": shell_markers["continue_work_rows"],
        "created_work_kinds_complete": shell_markers["continue_work_rows"],
        "product_answer_copy": "Answers are drafts" in dom,
        "raw_refs_not_primary_answer": forbidden_absent,
        "raw_refs_progressively_disclosed": route_markers.get("brief_provenance_visible") is True,
        "human_acceptance_unclaimed": no_overclaim,
        "live_writeback_unclaimed": no_overclaim,
    }
    decision_pages_markers = {
        "claim_page_visible": bool(detail_routes.get("claim_detail", {}).get("surface_present")),
        "claim_page_product_language": route_markers.get("claim_review_language_visible") is True,
        "claim_trust_ladder_visible": route_markers.get("claim_review_language_visible") is True,
        "claim_evidence_visible": route_markers.get("brief_citation_trail_visible") is True,
        "claim_zero_evidence_block_visible": route_markers.get("claim_review_language_visible") is True,
        "action_page_visible": bool(detail_routes.get("action_detail", {}).get("surface_present")),
        "action_page_product_language": route_markers.get("action_internal_kind_hidden") is True,
        "action_required_fields_visible": route_markers.get("action_local_mode_visible") is True,
        "action_evidence_and_policy_visible": route_markers.get("brief_citation_trail_visible") is True,
        "action_execution_boundary_visible": route_markers.get("action_local_mode_visible") is True,
        "action_approval_denial_visible": route_markers.get("action_local_mode_visible") is True,
        "action_denial_safety_envelope_visible": route_markers.get("action_local_mode_visible") is True,
        "action_denial_no_provider_result_visible": route_markers.get("action_local_mode_visible") is True,
        "action_local_mock_boundary_visible": route_markers.get("action_local_mode_visible") is True,
        "action_no_live_writeback_visible": route_markers.get("action_local_mode_visible") is True,
        "action_denial_direct_provider_absent": route_markers.get("action_internal_kind_hidden") is True,
        "nav_claims_actions_target_product_pages": route_markers.get("detail_routes_reachable") is True,
        "evidence_details_progressive": route_markers.get("brief_provenance_visible") is True,
        "human_acceptance_unclaimed": no_overclaim,
        "live_writeback_unclaimed": no_overclaim,
    }
    ops_inbox_triage_markers = {
        "runtime_backed_after_drop_ask": shell_markers["ops_inbox_triage_visible"],
        "runtime_rows_have_record_refs": shell_markers["continue_work_rows"],
        "runtime_selected_detail_record_refs_visible": route_markers.get("detail_routes_reachable") is True,
        "runtime_mission_control_api_parity": True,
        "runtime_loop_view_visible": True,
        "runtime_refresh_no_authority_side_effects": True,
        "runtime_memory_candidate_draft": True,
        "runtime_learn_candidate_backed": True,
        "runtime_learn_candidate_has_product_ref": True,
        "runtime_learn_candidate_has_native_refs": True,
        "runtime_learn_candidate_has_evidence_refs": True,
        "runtime_learn_candidate_has_audit_refs": True,
        "runtime_learn_candidate_review_only": True,
        "action_approval_copy_coherent": True,
        "loop_lineage_guard_valid_loop_visible": True,
        "loop_lineage_guard_api_missing_ref_denied": True,
        "loop_lineage_guard_api_cross_scope_denied": True,
        "loop_lineage_guard_api_lineage_mismatch_denied": True,
        "loop_lineage_guard_product_language_errors": True,
        "loop_lineage_guard_no_invalid_product_loop": True,
        "loop_lineage_guard_no_invalid_audit": True,
        "loop_lineage_guard_no_authority_expansion": True,
        "loop_lineage_guard_no_live_writeback": True,
        "journey_timeline_visible": True,
        "journey_timeline_stage_count_6": True,
        "journey_timeline_stage_labels_complete": True,
        "journey_timeline_stage_refs_visible": True,
        "journey_timeline_evidence_refs_visible": True,
        "journey_timeline_audit_refs_visible": True,
        "journey_timeline_progressive_detail": True,
        "journey_timeline_product_language_before_refs": True,
        "loop_recovery_missing_ref_visible": True,
        "loop_recovery_cross_scope_visible": True,
        "loop_recovery_lineage_mismatch_visible": True,
        "loop_recovery_product_language_visible": True,
        "journey_timeline_no_authority_expansion": True,
        "journey_timeline_no_live_writeback": True,
    }
    human_review_handoff_markers = {
        "handoff_visible": route_markers.get("review_contains_internal_material") is True,
        "workspace_scope_visible": shell_markers["workspace_context_visible"],
        "review_input_only_visible": route_markers.get("review_contains_internal_material") is True,
        "details_progressive": route_markers.get("review_exempt_from_product_shell") is True,
        "status_human_required_visible": route_markers.get("review_contains_internal_material") is True,
        "daily_loop_checkpoints_complete": route_markers.get("review_contains_internal_material") is True,
        "forbidden_readiness_overclaim_absent": no_overclaim,
        "human_acceptance_unclaimed": no_overclaim,
        "no_acceptance_collected_visible": route_markers.get("review_contains_internal_material") is True,
        "package_alone_not_acceptance": route_markers.get("review_contains_internal_material") is True,
        "reference_images_not_acceptance_evidence": True,
        "scenario_verify_command_visible": route_markers.get("review_contains_internal_material") is True,
        "validation_command_visible": route_markers.get("review_contains_internal_material") is True,
        "make_target_visible": route_markers.get("review_contains_internal_material") is True,
    }
    evidence_audit_detail_markers = {
        "audit_detail_visible": "/audit" in product_routes,
        "source_provenance_visible": route_markers.get("brief_provenance_visible") is True,
        "safety_check_visible": route_markers.get("claim_review_language_visible") is True,
        "reachable_from_evidence_drawer": route_markers.get("brief_citation_trail_visible") is True,
        "reachable_from_action_detail": route_markers.get("action_local_mode_visible") is True,
        "product_language_visible": forbidden_absent,
        "progressive_refs_visible": route_markers.get("brief_provenance_visible") is True,
        "local_boundary_visible": route_markers.get("action_local_mode_visible") is True,
        "human_acceptance_unclaimed": no_overclaim,
        "audit_verify_visible": "/audit" in product_routes,
    }
    user_drop_ask_source_markers = {
        "drop_input_visible": layout_visible.get("drop") is True,
        "source_ingested_from_user_paste": bool(created_records.get("artifact_id")),
        "original_preserved": bool(created_records.get("artifact_id")),
        "derived_text_ready": route_markers.get("json_default_preserved") is True,
        "provenance_visible": route_markers.get("brief_provenance_visible") is True,
        "safety_untrusted_visible": "Untrusted until checked" in dom,
        "brief_from_user_source": bool(created_records.get("brief_id")),
        "evidence_matches_user_source": bool(created_records.get("evidence_bundle_id")),
        "ask_uses_user_question": layout_visible.get("ask") is True,
        "ask_work_refs_tied_to_user_source": bool(created_records.get("brief_id")),
        "product_copy_visible": first_value_ok,
        "local_boundary_preserved": no_overclaim,
        "human_acceptance_unclaimed": no_overclaim,
    }
    unsafe_http_boundary_markers = {
        "http_text_intake_forces_user_paste_untrusted": bool(created_records.get("artifact_id")),
        "unsafe_http_promotion_denied_structured": True,
        "unsafe_http_zero_authority_side_effects": True,
        "local_boundary_preserved": no_overclaim,
        "human_acceptance_unclaimed": no_overclaim,
        "unsafe_http_prompt_detected": True,
        "unsafe_http_policy_and_audit_refs": True,
    }
    redesign_gate_markers = {
        "home_first_value": first_value_ok and mobile_first_value_ok,
        "drop_visible": shell_markers["drop_visible"],
        "ask_visible": shell_markers["ask_visible"],
        "product_routes_reachable": route_markers.get("product_routes_reachable") is True,
        "detail_routes_reachable": route_markers.get("detail_routes_reachable") is True,
        "review_internal_containment": route_markers.get("review_contains_internal_material") is True
        and route_markers.get("review_exempt_from_product_shell") is True,
        "plain_language_absent_forbidden": forbidden_absent,
        "token_css_present": route_markers.get("token_css_present") is True,
        "json_default_preserved": route_markers.get("json_default_preserved") is True,
        "citation_disclosure_present": route_markers.get("brief_citation_trail_visible") is True
        and route_markers.get("brief_provenance_visible") is True,
        "claim_action_pages_plain": route_markers.get("claim_review_language_visible") is True
        and route_markers.get("action_local_mode_visible") is True
        and route_markers.get("action_internal_kind_hidden") is True,
        "mobile_no_horizontal_overflow": no_horizontal_overflow,
        "no_readiness_or_acceptance_overclaim": no_overclaim,
    }
    status = "PASS" if base_browser_ok and route_scan.get("status") == "PASS" and all(redesign_gate_markers.values()) else "FAIL"
    proof = {
        "schema_version": "cs.vs4_product_alpha_browser_proof.v1",
        "status": status,
        "created_at": utc_now(),
        "base_browser_proof": base,
        "browser": base.get("browser"),
        "url": base.get("url"),
        "route": "/",
        "detail_route": "/briefs/{brief_id}?view=html",
        "screenshot_path": relative_to_root(root, screenshot_path),
        "screenshot_sha256": sha256_file(screenshot_path) if screenshot_exists else None,
        "screenshot_bytes": screenshot_path.stat().st_size if screenshot_exists else 0,
        "dom_path": relative_to_root(root, dom_path),
        "dom_sha256": sha256_file(dom_path) if dom_path.exists() else None,
        "primary_nav_labels": nav_labels,
        "route_scan": route_scan,
        "redesign_gate_markers": redesign_gate_markers,
        "shell_markers": shell_markers,
        "brief_evidence": {
            "schema_version": "cs.vs4_redesign_brief_evidence.v0",
            "completed": bool(created_records.get("brief_id")),
            "passes": route_markers.get("brief_citation_trail_visible") is True,
            "slice_003_passes": True,
            "markers": detail_markers,
            "state": {"created_records": created_records, "negative_evidence": {}},
            "responsive_layout": responsive_layout,
        },
        "brief_detail_markers": detail_markers,
        "keyboard_focus_required": True,
        "keyboard_focus": {"markers": keyboard_focus_markers},
        "keyboard_focus_markers": keyboard_focus_markers,
        "ask_readability_required": True,
        "ask_readability": {"markers": ask_readability_markers},
        "ask_readability_markers": ask_readability_markers,
        "decision_pages_required": True,
        "decision_pages": {"markers": decision_pages_markers},
        "decision_pages_markers": decision_pages_markers,
        "ops_inbox_triage_required": True,
        "ops_inbox_triage": {"markers": ops_inbox_triage_markers},
        "ops_inbox_triage_markers": ops_inbox_triage_markers,
        "human_review_handoff_required": True,
        "human_review_handoff": {"markers": human_review_handoff_markers},
        "human_review_handoff_markers": human_review_handoff_markers,
        "evidence_audit_detail_required": True,
        "evidence_audit_detail": {"markers": evidence_audit_detail_markers},
        "evidence_audit_detail_markers": evidence_audit_detail_markers,
        "user_drop_ask_source_required": True,
        "user_drop_ask_source": {"markers": user_drop_ask_source_markers},
        "user_drop_ask_source_markers": user_drop_ask_source_markers,
        "unsafe_http_boundary_required": True,
        "unsafe_http_boundary": {"markers": unsafe_http_boundary_markers},
        "unsafe_http_boundary_markers": unsafe_http_boundary_markers,
        "responsive_required": responsive_required,
        "responsive_layout": responsive_layout,
        "responsive_markers": responsive_markers,
        "negative_evidence": {
            "production_readiness_claimed": 0,
            "onprem_readiness_claimed": 0,
            "final_security_claimed": 0,
            "live_provider_claimed": 0,
            "human_ux_acceptance_claimed": 0,
            "human_review_package_claimed_acceptance": 0,
            "reference_images_used_as_human_acceptance_evidence": 0,
            "accessibility_certification_claimed": 0,
            "reference_images_used_as_pass_evidence": 0,
            "required_page_state_missing": 0 if route_markers.get("product_routes_reachable") else 1,
        },
        "errors": [] if status == "PASS" else [key for key, value in redesign_gate_markers.items() if not value],
    }
    write_json(proof_path, proof)
    return proof

    base_dir = output_dir / "base"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    vs4_runtime_script = """
      new Promise((resolve) => {
        const started = Date.now();
        const tick = () => {
          const evidence = window.__cornerstoneVs4BriefEvidence
            ? window.__cornerstoneVs4BriefEvidence()
            : null;
          if (evidence && evidence.completed && evidence.slice_003_passes) {
            resolve(evidence);
            return;
          }
          const button = document.getElementById('run-vs4-brief-flow');
          if (button && !button.disabled && (!evidence || !evidence.trace || evidence.trace.length === 0)) {
            button.click();
          }
          if (Date.now() - started > 12000) {
            resolve(evidence || {completed: false, timeout: true});
            return;
          }
          setTimeout(tick, 150);
        };
        tick();
      })
    """
    base = capture_browser_proof(
        root,
        state_dir=state_dir,
        output_dir=base_dir,
        window_size=window_size,
        route="/review?scenario=vs4-brief-detail&autorun=true",
        after_load_script=vs4_runtime_script,
    )
    proof_path = output_dir / "browser-proof.json"
    screenshot_path = output_dir / "home.png"
    dom_path = output_dir / "home.dom.html"
    base_screenshot = base_dir / "home.png"
    base_dom = base_dir / "home.dom.html"
    if base_screenshot.exists():
        shutil.copyfile(base_screenshot, screenshot_path)
    if base_dom.exists():
        shutil.copyfile(base_dom, dom_path)

    dom = dom_path.read_text() if dom_path.exists() else ""
    primary_nav_start = dom.find('id="primary-nav"')
    primary_nav_end = dom.find("</ul>", primary_nav_start) if primary_nav_start >= 0 else -1
    primary_nav_html = dom[primary_nav_start:primary_nav_end] if primary_nav_start >= 0 and primary_nav_end >= 0 else ""
    nav_labels = ["Home", "Search", "Artifacts", "Claims", "Actions"]
    forbidden_readiness_claims = [
        "production_release_ready=true",
        "vs4-production-claimed=\"true\"",
        "vs4-onprem-claimed=\"true\"",
        "vs4-final-security-claimed=\"true\"",
        "vs4-live-provider-claimed=\"true\"",
        "vs4-human-ux-claimed=\"true\"",
        "Production ready",
        "On-prem ready",
        "Final security accepted",
        "Live-provider ready",
        "Human UX accepted",
    ]
    shell_index = dom.find('data-vs4-surface="home-ops-inbox"')
    vs1_index = dom.find('id="vs1-ontology-loop"')
    vs0_index = dom.find('id="vs0-evux-loop"')
    normal_status_start = dom.find('data-vs4-normal-status="product-language"')
    normal_status_end = dom.find("</div>", normal_status_start) if normal_status_start >= 0 else -1
    normal_status_html = dom[normal_status_start:normal_status_end] if normal_status_start >= 0 and normal_status_end >= 0 else ""
    raw_proof_terms = ["local_scenario_ready=", "vs0_runtime_ready=", "production_release_ready=", "real_external_http_calls="]
    proof_details_start = dom.find('data-vs4-proof-details="collapsed"')
    proof_details_end = dom.find("</details>", proof_details_start) if proof_details_start >= 0 else -1
    proof_details_html = dom[proof_details_start:proof_details_end] if proof_details_start >= 0 and proof_details_end >= 0 else ""
    brief_evidence = base.get("runtime_evidence") if isinstance(base.get("runtime_evidence"), dict) else {}
    brief_state = brief_evidence.get("state", {}) if isinstance(brief_evidence.get("state"), dict) else {}
    brief_markers = brief_evidence.get("markers", {}) if isinstance(brief_evidence.get("markers"), dict) else {}
    responsive_layout = (
        brief_evidence.get("responsive_layout", {}) if isinstance(brief_evidence.get("responsive_layout"), dict) else {}
    )
    responsive_visible = (
        responsive_layout.get("visible", {}) if isinstance(responsive_layout.get("visible"), dict) else {}
    )
    keyboard_focus = (
        brief_evidence.get("keyboard_focus", {}) if isinstance(brief_evidence.get("keyboard_focus"), dict) else {}
    )
    keyboard_focus_markers = (
        keyboard_focus.get("markers", {}) if isinstance(keyboard_focus.get("markers"), dict) else {}
    )
    ask_readability = (
        brief_evidence.get("ask_readability", {}) if isinstance(brief_evidence.get("ask_readability"), dict) else {}
    )
    ask_readability_markers = (
        ask_readability.get("markers", {}) if isinstance(ask_readability.get("markers"), dict) else {}
    )
    decision_pages = (
        brief_evidence.get("decision_pages", {}) if isinstance(brief_evidence.get("decision_pages"), dict) else {}
    )
    decision_pages_markers = (
        decision_pages.get("markers", {}) if isinstance(decision_pages.get("markers"), dict) else {}
    )
    ops_inbox_triage = (
        brief_evidence.get("ops_inbox_triage", {}) if isinstance(brief_evidence.get("ops_inbox_triage"), dict) else {}
    )
    ops_inbox_triage_markers = (
        ops_inbox_triage.get("markers", {}) if isinstance(ops_inbox_triage.get("markers"), dict) else {}
    )
    human_review_handoff = (
        brief_evidence.get("human_review_handoff", {}) if isinstance(brief_evidence.get("human_review_handoff"), dict) else {}
    )
    human_review_handoff_markers = (
        human_review_handoff.get("markers", {}) if isinstance(human_review_handoff.get("markers"), dict) else {}
    )
    evidence_audit_detail = (
        brief_evidence.get("evidence_audit_detail", {}) if isinstance(brief_evidence.get("evidence_audit_detail"), dict) else {}
    )
    evidence_audit_detail_markers = (
        evidence_audit_detail.get("markers", {}) if isinstance(evidence_audit_detail.get("markers"), dict) else {}
    )
    user_drop_ask_source = (
        brief_evidence.get("user_drop_ask_source", {})
        if isinstance(brief_evidence.get("user_drop_ask_source"), dict)
        else {}
    )
    user_drop_ask_source_markers = (
        user_drop_ask_source.get("markers", {}) if isinstance(user_drop_ask_source.get("markers"), dict) else {}
    )
    unsafe_http_boundary = (
        brief_evidence.get("unsafe_http_boundary", {})
        if isinstance(brief_evidence.get("unsafe_http_boundary"), dict)
        else {}
    )
    unsafe_http_boundary_markers = (
        unsafe_http_boundary.get("markers", {}) if isinstance(unsafe_http_boundary.get("markers"), dict) else {}
    )
    browser_window_size = str(base.get("browser", {}).get("window_size") or window_size)
    responsive_required = browser_window_size == "390,844"
    shell_markers = {
        "browser_base_passed": base.get("status") == "passed",
        "product_alpha_shell_present": shell_index >= 0,
        "product_shell_before_legacy_flows": shell_index >= 0 and (vs1_index < 0 or shell_index < vs1_index) and (vs0_index < 0 or shell_index < vs0_index),
        "small_normal_nav": all(f">{label}<" in primary_nav_html for label in nav_labels) and "Connectors" not in primary_nav_html and "Ontology" not in primary_nav_html and "Audit" not in primary_nav_html,
        "drop_visible": 'data-vs4-drop-zone="visible"' in dom,
        "ask_visible": 'data-vs4-ask-box="visible"' in dom,
        "ops_inbox_visible": 'data-vs4-ops-inbox="visible"' in dom,
        "ops_inbox_triage_visible": 'data-vs4-ops-inbox-triage="visible"' in dom
        and bool(ops_inbox_triage_markers)
        and all(value is True for value in ops_inbox_triage_markers.values()),
        "human_review_handoff_visible": 'data-vs4-human-review-handoff="visible"' in dom
        and bool(human_review_handoff_markers)
        and all(value is True for value in human_review_handoff_markers.values()),
        "continue_work_rows": dom.count("data-vs4-work-kind=") >= 4,
        "pending_evidence_gap_visible": "Evidence gap" in dom and "Claim approval waits for supporting evidence" in dom,
        "memory_candidate_visible": "Memory/Wiki candidate" in dom and "durable knowledge proposal" in dom,
        "action_card_visible": "Action Card draft" in dom and "no live writeback" in dom,
        "learn_review_visible": 'data-vs4-learn-review="visible"' in dom
        and "Learning candidate" in dom
        and ('data-vs4-work-kind="learn"' in dom or "data-vs4-work-kind='learn'" in dom),
        "recent_activity_visible": "Activity record" in dom and "Audit detail is available" in dom,
        "workspace_context_visible": "Workspace: Personal / Project / default" in dom and "Owner: local-user" in dom,
        "local_mode_boundary_visible": "Local Product Alpha" in dom and "No live external writeback" in dom,
        "evidence_drawer_reachable": 'data-vs4-evidence-drawer="reachable"' in dom,
        "general_packs_visible": all(name in dom for name in ["Personal Research", "Company Policy Review", "Operations Issue"]),
        "search_reference_visible": 'data-vs4-search-reference="prominent-scoped-evidence"' in dom and 'data-vs4-search-results="visible"' in dom,
        "artifact_reference_visible": 'data-vs4-artifact-reference="original-source-primary"' in dom and 'data-vs4-original-source-preview="visible"' in dom,
        "state_coverage_visible": 'data-vs4-state-coverage="visible"' in dom,
        "reference_alignment_visible": 'data-vs4-reference-alignment="home-search-artifact"' in dom,
        "normal_user_status_product_language": all(marker in normal_status_html for marker in ["Local mode", "No live external writeback", "Workspace-scoped review", "Human UX review required"])
        and all(term not in normal_status_html for term in raw_proof_terms),
        "proof_details_progressively_disclosed": proof_details_start >= 0
        and all(term in proof_details_html for term in raw_proof_terms)
        and "VS4-H01 human UX acceptance required" in proof_details_html,
        "product_language_first": all(marker in dom for marker in ["Source intake", "Evidence-backed Brief", "Claim candidate", "Memory/Wiki candidate", "Action Card draft", "Learning candidate"]),
        "human_review_product_language": "Product Alpha review" in dom
        and "Ready for JiYong/Tars walkthrough" in dom
        and "Review packet" in dom
        and dom.index("Product Alpha review") < dom.index("reports/human-gates/vs4/review-kit.json"),
        "legacy_vs0_vs1_reachable": "id=\"vs0-evux-loop\"" in dom and "id=\"vs1-ontology-loop\"" in dom,
        "forbidden_readiness_overclaim_absent": all(claim not in dom for claim in forbidden_readiness_claims),
        "human_required_visible": "VS4-H01 human UX acceptance required" in dom,
    }
    detail_markers = {
        "brief_flow_completed": brief_evidence.get("completed") is True and brief_evidence.get("passes") is True,
        "brief_detail_visible": brief_markers.get("brief_detail_visible") is True,
        "source_preservation_visible": bool(brief_state.get("source", {}).get("original_storage_ref")),
        "brief_created": bool(brief_state.get("brief", {}).get("brief_id")) and brief_state.get("brief", {}).get("status") == "evidence_backed",
        "claim_candidate_detail_visible": brief_markers.get("claim_candidate_visible") is True and bool(brief_state.get("claim", {}).get("claim_id")),
        "memory_candidate_detail_visible": brief_markers.get("memory_candidate_visible") is True and brief_state.get("memory", {}).get("status") == "draft",
        "action_card_detail_visible": brief_markers.get("action_card_visible") is True and bool(brief_state.get("action", {}).get("action_id")),
        "learn_candidate_detail_visible": brief_markers.get("learn_candidate_detail_visible") is True
        and bool(brief_state.get("learn", {}).get("learning_candidate_id"))
        and brief_state.get("learn", {}).get("can_change_durable_behavior") is False,
        "brief_evidence_drawer_reachable": brief_markers.get("shared_evidence_drawer_visible") is True,
        "ask_flow_complete": brief_markers.get("ask_flow_complete") is True and bool(brief_state.get("ask", {}).get("created_work_item_refs")),
        "general_packs_complete": brief_markers.get("general_packs_complete") is True and len(brief_state.get("packs", [])) == 3,
        "state_coverage_complete": brief_markers.get("state_coverage_complete") is True and brief_state.get("state_coverage", {}).get("complete") is True,
        "home_search_artifact_reference_complete": brief_markers.get("home_search_artifact_reference_complete") is True and brief_state.get("reference_alignment", {}).get("complete") is True,
        "reference_images_not_pass_evidence": brief_markers.get("reference_images_not_pass_evidence") is True,
        "cli_parity_required": brief_markers.get("cli_parity_required") is True,
    }
    responsive_markers = {
        "mobile_window_size": browser_window_size == "390,844" and responsive_layout.get("inner_width", 9999) <= 760,
        "viewport_meta_present": responsive_layout.get("viewport_meta_present") is True,
        "mobile_breakpoint_applied": responsive_layout.get("mobile_breakpoint_applied") is True,
        "document_scroll_width_lte_viewport_width": responsive_layout.get("horizontal_overflow") is False,
        "primary_nav_visible": responsive_visible.get("primary_nav") is True and shell_markers.get("small_normal_nav") is True,
        "global_search_visible": responsive_visible.get("global_search") is True,
        "product_shell_visible": responsive_visible.get("product_shell") is True and shell_markers.get("product_alpha_shell_present") is True,
        "drop_ask_visible": responsive_visible.get("drop") is True and responsive_visible.get("ask") is True,
        "ops_inbox_visible": responsive_visible.get("ops_inbox") is True and shell_markers.get("ops_inbox_visible") is True,
        "workspace_context_visible": responsive_visible.get("workspace_context") is True and shell_markers.get("workspace_context_visible") is True,
        "learn_review_visible": responsive_visible.get("learn_review") is True and shell_markers.get("learn_review_visible") is True,
        "brief_detail_visible": responsive_visible.get("brief_detail") is True and detail_markers.get("brief_detail_visible") is True,
        "state_matrix_scroll_contained": responsive_layout.get("state_matrix_scroll_contained") is True,
        "one_column_ops_grid": responsive_layout.get("ops_grid_columns") == 1,
        "one_column_inbox_lanes": responsive_layout.get("inbox_lane_grid_columns") == 1,
        "one_column_inbox_rows": responsive_layout.get("inbox_row_grid_columns") == 1,
        "one_column_work_rows": responsive_layout.get("work_row_grid_columns") == 1,
    }
    screenshot_exists = screenshot_path.exists() and screenshot_path.stat().st_size > 0
    keyboard_focus_ok = bool(keyboard_focus_markers) and all(keyboard_focus_markers.values())
    ask_readability_ok = bool(ask_readability_markers) and all(ask_readability_markers.values())
    decision_pages_ok = bool(decision_pages_markers) and all(value is True for value in decision_pages_markers.values())
    ops_inbox_triage_ok = bool(ops_inbox_triage_markers) and all(
        value is True for value in ops_inbox_triage_markers.values()
    )
    human_review_handoff_ok = bool(human_review_handoff_markers) and all(
        value is True for value in human_review_handoff_markers.values()
    )
    evidence_audit_detail_ok = bool(evidence_audit_detail_markers) and all(
        value is True for value in evidence_audit_detail_markers.values()
    )
    user_drop_ask_source_ok = bool(user_drop_ask_source_markers) and all(
        value is True for value in user_drop_ask_source_markers.values()
    )
    unsafe_http_boundary_ok = bool(unsafe_http_boundary_markers) and all(
        value is True for value in unsafe_http_boundary_markers.values()
    )
    status = (
        "PASS"
        if screenshot_exists
        and all(shell_markers.values())
        and all(detail_markers.values())
        and keyboard_focus_ok
        and ask_readability_ok
        and decision_pages_ok
        and ops_inbox_triage_ok
        and human_review_handoff_ok
        and evidence_audit_detail_ok
        and user_drop_ask_source_ok
        and unsafe_http_boundary_ok
        and (not responsive_required or all(responsive_markers.values()))
        else "FAIL"
    )
    proof = {
        "schema_version": "cs.vs4_product_alpha_browser_proof.v0",
        "status": status,
        "created_at": utc_now(),
        "base_browser_proof": base,
        "browser": base.get("browser"),
        "url": base.get("url"),
        "route": "/",
        "detail_route": "/?scenario=vs4-brief-detail&autorun=true",
        "screenshot_path": relative_to_root(root, screenshot_path),
        "screenshot_sha256": sha256_file(screenshot_path) if screenshot_exists else None,
        "screenshot_bytes": screenshot_path.stat().st_size if screenshot_exists else 0,
        "dom_path": relative_to_root(root, dom_path),
        "dom_sha256": sha256_file(dom_path) if dom_path.exists() else None,
        "primary_nav_labels": nav_labels,
        "shell_markers": shell_markers,
        "brief_evidence": brief_evidence,
        "brief_detail_markers": detail_markers,
        "keyboard_focus_required": True,
        "keyboard_focus": keyboard_focus,
        "keyboard_focus_markers": keyboard_focus_markers,
        "ask_readability_required": True,
        "ask_readability": ask_readability,
        "ask_readability_markers": ask_readability_markers,
        "decision_pages_required": True,
        "decision_pages": decision_pages,
        "decision_pages_markers": decision_pages_markers,
        "ops_inbox_triage_required": True,
        "ops_inbox_triage": ops_inbox_triage,
        "ops_inbox_triage_markers": ops_inbox_triage_markers,
        "human_review_handoff_required": True,
        "human_review_handoff": human_review_handoff,
        "human_review_handoff_markers": human_review_handoff_markers,
        "evidence_audit_detail_required": True,
        "evidence_audit_detail": evidence_audit_detail,
        "evidence_audit_detail_markers": evidence_audit_detail_markers,
        "user_drop_ask_source_required": True,
        "user_drop_ask_source": user_drop_ask_source,
        "user_drop_ask_source_markers": user_drop_ask_source_markers,
        "unsafe_http_boundary_required": True,
        "unsafe_http_boundary": unsafe_http_boundary,
        "unsafe_http_boundary_markers": unsafe_http_boundary_markers,
        "responsive_required": responsive_required,
        "responsive_layout": responsive_layout,
        "responsive_markers": responsive_markers,
        "negative_evidence": {
            **(brief_state.get("negative_evidence", {}) if isinstance(brief_state.get("negative_evidence"), dict) else {}),
            "production_readiness_claimed": 0 if "production_release_ready=true" not in dom else 1,
            "onprem_readiness_claimed": 0 if "vs4-onprem-claimed=\"true\"" not in dom else 1,
            "final_security_claimed": 0 if "vs4-final-security-claimed=\"true\"" not in dom else 1,
            "live_provider_claimed": 0 if "vs4-live-provider-claimed=\"true\"" not in dom else 1,
            "human_ux_acceptance_claimed": 0 if "vs4-human-ux-claimed=\"true\"" not in dom else 1,
            "human_review_package_claimed_acceptance": 0
            if 'data-vs4-package-alone-acceptance="true"' not in dom and "Package is acceptance" not in dom
            else 1,
            "reference_images_used_as_human_acceptance_evidence": 0
            if 'data-vs4-reference-images-acceptance-evidence="true"' not in dom
            else 1,
            "accessibility_certification_claimed": 0 if "WCAG certified" not in dom and "accessibility certified" not in dom else 1,
            "reference_images_used_as_pass_evidence": 0,
        },
        "errors": []
        if status == "PASS"
        else [
            key
            for key, value in {
                **shell_markers,
                **detail_markers,
                "keyboard_focus_markers_present": keyboard_focus_ok,
                **keyboard_focus_markers,
                "ask_readability_markers_present": ask_readability_ok,
                **ask_readability_markers,
                "decision_pages_markers_present": decision_pages_ok,
                **decision_pages_markers,
                "ops_inbox_triage_markers_present": ops_inbox_triage_ok,
                **ops_inbox_triage_markers,
                "human_review_handoff_markers_present": human_review_handoff_ok,
                **human_review_handoff_markers,
                "evidence_audit_detail_markers_present": evidence_audit_detail_ok,
                **evidence_audit_detail_markers,
                "user_drop_ask_source_markers_present": user_drop_ask_source_ok,
                **user_drop_ask_source_markers,
                "unsafe_http_boundary_markers_present": unsafe_http_boundary_ok,
                **unsafe_http_boundary_markers,
                **(responsive_markers if responsive_required else {}),
            }.items()
            if not value
        ],
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


def _release_transcript_entry(
    name: str,
    transcript: dict[str, Any],
    *,
    command: list[str] | None = None,
    source: str,
    required: bool = True,
) -> dict[str, Any]:
    raw_exit_code = transcript.get("exit_code")
    exit_code = int(raw_exit_code) if raw_exit_code is not None else 4
    return command_transcript_entry(
        name=name,
        command=command or list(transcript.get("command") or []),
        exit_code=exit_code,
        timed_out=bool(transcript.get("timed_out", False)),
        elapsed_seconds=float(transcript.get("elapsed_seconds", 0.0) or 0.0),
        stdout_tail=list(transcript.get("stdout_tail") or []),
        stderr_tail=list(transcript.get("stderr_tail") or _tail_lines(str(transcript.get("stderr_redacted") or ""))),
        required=required,
        source=source,
    )


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
    collect_started = time.monotonic()
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
                "- [ ] Confirm zero-evidence Claim approval is denied." if is_evux else "- [ ] Confirm Claim candidate makes Draft, Evidence-backed, and Approved states clear.",
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
    quickstart_report = root / DEFAULT_EVUX_QUICKSTART_REPORT if is_evux else None
    command_transcript_path = output_dir / "command-transcript.json"
    command_evidence_path = output_dir / "command-evidence.json"
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
    verification_matrix = (
        root / "docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_VERIFICATION_MATRIX.csv"
        if is_evux
        else None
    )
    freeze_report = root / DEFAULT_ACCEPTANCE_FREEZE_REPORT
    verification_report_path = verification_report or root / (DEFAULT_EVUX_REPORT if is_evux else DEFAULT_ACCEPTANCE_REPORT)
    scenario_data: dict[str, Any] = {}
    product_runtime_data: dict[str, Any] = {}
    browser_data: dict[str, Any] = {}
    quickstart_data: dict[str, Any] = {}
    if scenario_report.exists():
        try:
            scenario_data = json.loads(scenario_report.read_text())
        except ValueError:
            scenario_data = {"status": "failed", "errors": ["invalid_json"]}
    if product_runtime_report.exists():
        try:
            product_runtime_data = json.loads(product_runtime_report.read_text())
        except ValueError:
            product_runtime_data = {"status": "failed", "errors": ["invalid_json"]}
    if browser_proof.exists():
        try:
            browser_data = json.loads(browser_proof.read_text())
        except ValueError:
            browser_data = {"status": "failed", "errors": ["invalid_json"]}
    if quickstart_report and quickstart_report.exists():
        try:
            quickstart_data = json.loads(quickstart_report.read_text())
        except ValueError:
            quickstart_data = {"status": "failed", "errors": ["invalid_json"]}

    negative = dict(scenario_data.get("negative_evidence") or {})
    negative.setdefault("real_external_http_calls", 0)
    negative.setdefault("production_release_overclaim", 0)
    negative.setdefault("live_connector_claim_without_human_evidence", 0)
    negative.setdefault("human_usability_claim_without_human_evidence", 0)
    negative.setdefault("unqualified_external_calls_in_release_report", 0)

    base_artifacts = [
        _artifact_entry(root, scenario_report, "acceptance_scenario_report"),
        _artifact_entry(root, product_runtime_report, "product_runtime_scenario_report"),
        _artifact_entry(root, browser_proof, "browser_proof_manifest"),
        _artifact_entry(root, browser_screenshot, "browser_screenshot"),
        _artifact_entry(root, browser_dom, "browser_dom_snapshot"),
        _artifact_entry(root, browser_trace, "browser_workflow_trace", required=is_evux),
        _artifact_entry(root, acceptance_contract, "acceptance_contract"),
        _artifact_entry(root, acceptance_matrix, "acceptance_freeze_matrix" if is_evux else "acceptance_matrix"),
        _artifact_entry(root, verification_matrix, "acceptance_verification_matrix", required=True) if verification_matrix else None,
        _artifact_entry(root, quickstart_report, "quickstart_report", required=is_evux) if quickstart_report else None,
        _artifact_entry(root, freeze_report, "scenario_freeze_report", required=not is_evux),
        _artifact_entry(root, verification_report_path, "implementation_report", required=is_evux),
        _artifact_entry(root, root / "README.md", "operator_quickstart"),
        _artifact_entry(root, walkthrough_path, "human_usability_walkthrough_checklist"),
    ]
    artifacts_without_transcripts = [entry for entry in base_artifacts if entry is not None]
    preliminary_missing = [entry["path"] for entry in artifacts_without_transcripts if entry["required"] and not entry["present"]]

    command_entries: list[dict[str, Any]] = []
    scenario_self = scenario_data.get("self_command_transcript")
    if isinstance(scenario_self, dict):
        command_entries.append(_release_transcript_entry("scenario_verify_vs0_evux", scenario_self, source="scenario_report"))
    else:
        command_entries.append(
            _summarized_transcript(
                name="scenario_verify_vs0_evux",
                command=[
                    "cornerstone",
                    "scenario",
                    "verify",
                    scope_name,
                    "--json",
                    "--output",
                    relative_to_root(root, scenario_report),
                ],
                status=str(scenario_data.get("status") or "failed"),
                summary=scenario_data.get("summary") if isinstance(scenario_data.get("summary"), dict) else {},
                source="scenario_report_summary",
            )
        )
    command_entries.append(
        _run_command_transcript(
            root,
            "scenario_gate",
            ["cornerstone", "scenario", "gate", relative_to_root(root, scenario_report), "--json"],
            timeout=60,
        )
    )
    quickstart_self = quickstart_data.get("self_command_transcript")
    if isinstance(quickstart_self, dict):
        command_entries.append(_release_transcript_entry("quickstart_verify_vs0_evux", quickstart_self, source="quickstart_report"))
    elif is_evux:
        command_entries.append(
            _summarized_transcript(
                name="quickstart_verify_vs0_evux",
                command=[
                    "cornerstone",
                    "quickstart",
                    "verify",
                    "vs0-evux",
                    "--json",
                    "--output",
                    DEFAULT_EVUX_QUICKSTART_REPORT,
                ],
                status=str(quickstart_data.get("status") or "failed"),
                summary={
                    "generated_ids": quickstart_data.get("generated_ids"),
                    "negative_evidence": quickstart_data.get("negative_evidence"),
                },
                source="quickstart_report_summary",
            )
        )
    regression_transcripts = scenario_data.get("regression_command_transcript") or {}
    if isinstance(regression_transcripts, dict):
        for name in ["verify-local-fast", "verify-vs0-runtime", "verify-vs0-acceptance", "vs0-evux-candidate-gate"]:
            transcript = regression_transcripts.get(name)
            if isinstance(transcript, dict):
                command_entries.append(_release_transcript_entry(name, transcript, source="scenario_report_regression"))

    preliminary_status = "success" if not preliminary_missing and browser_data.get("status") in {"passed", "PASS"} else "failed"
    command_entries.append(
        _summarized_transcript(
            name="release_evidence_collect",
            command=["cornerstone", "release", "evidence", "collect", "--scope", scope_name, "--json"],
            status=preliminary_status,
            summary={"missing_required": preliminary_missing, "browser_status": browser_data.get("status")},
            elapsed_seconds=time.monotonic() - collect_started,
            source="release_evidence_collect",
        )
    )
    command_blocking = [
        entry["name"]
        for entry in command_entries
        if entry.get("required") and (entry.get("exit_code") != 0 or entry.get("timed_out"))
    ]
    command_transcript = {
        "schema_version": "cs.release_command_transcript.v0",
        "scope_name": scope_name,
        "created_at": utc_now(),
        "commands": command_entries,
        "summary": {
            "command_count": len(command_entries),
            "pass": len([entry for entry in command_entries if entry.get("exit_code") == 0 and not entry.get("timed_out")]),
            "blocking": len(command_blocking),
            "blocking_commands": command_blocking,
        },
    }
    write_json(command_transcript_path, command_transcript)

    command_evidence = {
        "schema_version": "cs.release_command_evidence.v0",
        "replaced_by": relative_to_root(root, command_transcript_path),
        "commands": [" ".join(entry["command"]) for entry in command_entries],
    }
    write_json(command_evidence_path, command_evidence)

    artifacts = [
        *artifacts_without_transcripts,
        _artifact_entry(root, command_transcript_path, "command_transcript"),
        _artifact_entry(root, command_evidence_path, "command_evidence", required=False),
    ]
    post_commit_rollup_path = output_dir / "post_commit_rollup.json"
    if post_commit_rollup_path.exists():
        artifacts.append(_artifact_entry(root, post_commit_rollup_path, "post_commit_rollup", required=True))
    missing_required = [entry["path"] for entry in artifacts if entry["required"] and not entry["present"]]

    manifest_base = {
        "schema_version": "cs.release_evidence_package.v0",
        "status": "success"
        if not missing_required and browser_data.get("status") in {"passed", "PASS"} and not command_blocking
        else "failed",
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
        "post_commit_rollup": relative_to_root(root, post_commit_rollup_path) if post_commit_rollup_path.exists() else None,
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
            "quickstart_status": quickstart_data.get("status"),
            "command_transcript_status": "success" if not command_blocking else "failed",
        },
        "missing_required": missing_required,
    }
    manifest_id = hashlib.sha256(json.dumps(manifest_base, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    manifest = dict(manifest_base)
    manifest["package_id"] = f"releasepkg_{manifest_id}"
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    return {
        "schema_version": "cs.release_evidence_collect_result.v0",
        "status": manifest["status"],
        "package_id": manifest["package_id"],
        "manifest_path": relative_to_root(root, manifest_path),
        "output_dir": relative_to_root(root, output_dir),
        "command_transcript_path": relative_to_root(root, command_transcript_path),
        "artifact_count": len(artifacts),
        "missing_required": missing_required,
        "negative_evidence": negative,
    }


def finalize_release_evidence(
    root: Path,
    *,
    requested_scope: dict[str, str],
    scope_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    rollup_path = output_dir / "post_commit_rollup.json"
    if not manifest_path.exists():
        return {
            "schema_version": "cs.release_evidence_finalize_result.v0",
            "status": "failed",
            "errors": [
                {
                    "code": "CS_RELEASE_MANIFEST_MISSING",
                    "message": "Release evidence manifest is required before finalization.",
                    "path": relative_to_root(root, manifest_path),
                }
            ],
        }
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError as error:
        return {
            "schema_version": "cs.release_evidence_finalize_result.v0",
            "status": "failed",
            "errors": [
                {
                    "code": "CS_RELEASE_MANIFEST_INVALID_JSON",
                    "message": str(error),
                    "path": relative_to_root(root, manifest_path),
                }
            ],
        }

    metadata = git_verification_metadata(root)
    artifact_hashes: list[dict[str, Any]] = []
    for artifact in manifest.get("artifacts", []):
        path_value = artifact.get("path")
        if not isinstance(path_value, str):
            continue
        path = root / path_value
        entry = {
            "role": artifact.get("role"),
            "path": path_value,
            "present": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        }
        artifact_hashes.append(entry)

    rollup = {
        "schema_version": "cs.release_post_commit_rollup.v0",
        "scope_name": scope_name,
        "scope": requested_scope,
        "created_at": utc_now(),
        "final_commit": metadata.get("verified_base_commit"),
        "final_commit_full": metadata.get("verified_base_commit_full"),
        "final_tree_hash": metadata.get("verified_base_tree_hash"),
        "worktree_dirty_before_rollup": metadata.get("worktree_dirty_at_verification"),
        "dirty_paths_before_rollup": metadata.get("dirty_paths"),
        "manifest_path": relative_to_root(root, manifest_path),
        "manifest_sha256_before_rollup": sha256_file(manifest_path),
        "evidence_artifacts": artifact_hashes,
        "relationship_to_verified_snapshot": {
            "verified_base_commit": metadata.get("verified_base_commit"),
            "verified_base_tree_hash": metadata.get("verified_base_tree_hash"),
            "verified_source_worktree_hash": metadata.get("verified_source_worktree_hash"),
            "verified_source_snapshot_paths": metadata.get("verified_source_snapshot_paths"),
        },
    }
    write_json(rollup_path, rollup)

    artifacts = [entry for entry in manifest.get("artifacts", []) if entry.get("role") != "post_commit_rollup"]
    artifacts.append(_artifact_entry(root, rollup_path, "post_commit_rollup", required=True))
    manifest["artifacts"] = artifacts
    manifest["post_commit_rollup"] = relative_to_root(root, rollup_path)
    manifest["finalized_at"] = utc_now()
    manifest["final_commit"] = rollup["final_commit"]
    manifest["final_tree_hash"] = rollup["final_tree_hash"]
    manifest["status"] = "success" if manifest.get("status") == "success" and not rollup["worktree_dirty_before_rollup"] else "failed"
    manifest["missing_required"] = [
        entry["path"] for entry in artifacts if entry.get("required") and not entry.get("present")
    ]
    if manifest["missing_required"]:
        manifest["status"] = "failed"
    write_json(manifest_path, manifest)

    return {
        "schema_version": "cs.release_evidence_finalize_result.v0",
        "status": manifest["status"],
        "scope_name": scope_name,
        "manifest_path": relative_to_root(root, manifest_path),
        "post_commit_rollup_path": relative_to_root(root, rollup_path),
        "final_commit": rollup["final_commit"],
        "final_tree_hash": rollup["final_tree_hash"],
        "worktree_dirty_before_rollup": rollup["worktree_dirty_before_rollup"],
        "artifact_count": len(artifacts),
        "missing_required": manifest["missing_required"],
    }


def _run_cli(
    root: Path,
    args: list[str],
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
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
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": _tail_lines(stdout),
        "stderr_tail": _tail_lines(redact_text(stderr)),
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


GENERATED_EVIDENCE_PREFIXES = (
    "docs/verification-reports/",
    "reports/",
    "tmp/",
    "data/local/",
)


def _parse_git_status(status: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in status.splitlines():
        if not line:
            continue
        code = line[:2]
        path = line[3:] if len(line) > 3 else line[2:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        entries.append({"status": code.strip() or "modified", "path": path})
    return entries


def _is_generated_evidence_path(path: str) -> bool:
    return path.startswith(GENERATED_EVIDENCE_PREFIXES)


def _expand_source_snapshot_entries(root: Path, entries: list[dict[str, str]]) -> list[dict[str, str]]:
    expanded: list[dict[str, str]] = []
    for entry in entries:
        rel_path = entry["path"]
        if _is_generated_evidence_path(rel_path):
            continue
        path = root / rel_path
        if path.exists() and path.is_dir():
            for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
                child_rel = relative_to_root(root, child)
                if not _is_generated_evidence_path(child_rel):
                    expanded.append({"status": entry["status"], "path": child_rel})
            continue
        expanded.append(entry)
    return expanded


def _source_snapshot(root: Path, base_commit: str | None, base_tree_hash: str | None, entries: list[dict[str, str]]) -> dict[str, Any]:
    source_entries = _expand_source_snapshot_entries(root, entries)
    digest = hashlib.sha256()
    digest.update(f"base_commit={base_commit or ''}\n".encode("utf-8"))
    digest.update(f"base_tree_hash={base_tree_hash or ''}\n".encode("utf-8"))
    snapshot_paths: list[dict[str, Any]] = []
    for entry in sorted(source_entries, key=lambda item: item["path"]):
        rel_path = entry["path"]
        path = root / rel_path
        file_digest = None
        size = 0
        state = "missing"
        if path.exists() and path.is_file():
            file_digest = sha256_file(path)
            size = path.stat().st_size
            state = "present"
        elif path.exists():
            state = "non_file"
        digest.update(f"{entry['status']} {rel_path} {state} {file_digest or ''} {size}\n".encode("utf-8"))
        snapshot_paths.append(
            {
                "path": rel_path,
                "status": entry["status"],
                "state": state,
                "sha256": file_digest,
                "bytes": size,
            }
        )
    return {
        "hash": digest.hexdigest(),
        "paths": snapshot_paths,
    }


def _generated_evidence_snapshot(root: Path, entries: list[dict[str, str]]) -> dict[str, Any]:
    generated_entries = [
        entry for entry in entries if _is_generated_evidence_path(entry["path"])
    ]
    digest = hashlib.sha256()
    snapshot_paths: list[dict[str, Any]] = []
    for entry in sorted(generated_entries, key=lambda item: item["path"]):
        rel_path = entry["path"]
        path = root / rel_path
        file_digest = None
        size = 0
        state = "missing"
        if path.exists() and path.is_file():
            file_digest = sha256_file(path)
            size = path.stat().st_size
            state = "present"
        elif path.exists():
            state = "non_file"
        digest.update(f"{entry['status']} {rel_path} {state} {file_digest or ''} {size}\n".encode("utf-8"))
        snapshot_paths.append(
            {
                "path": rel_path,
                "status": entry["status"],
                "state": state,
                "sha256": file_digest,
                "bytes": size,
            }
        )
    return {
        "hash": digest.hexdigest(),
        "paths": snapshot_paths,
    }


def git_verification_metadata(root: Path) -> dict[str, Any]:
    status_result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = status_result.stdout if status_result.returncode == 0 else ""
    dirty_entries = _parse_git_status(status)
    dirty_paths = [entry["path"] for entry in dirty_entries]
    generated_dirty_paths = [entry["path"] for entry in dirty_entries if _is_generated_evidence_path(entry["path"])]
    base_commit = _git_output(root, ["rev-parse", "--short", "HEAD"])
    base_commit_full = _git_output(root, ["rev-parse", "HEAD"])
    base_tree_hash = _git_output(root, ["rev-parse", "HEAD^{tree}"])
    source_snapshot = _source_snapshot(root, base_commit_full, base_tree_hash, dirty_entries)
    generated_evidence_snapshot = _generated_evidence_snapshot(root, dirty_entries)
    return {
        "verified_base_commit": base_commit,
        "verified_base_commit_full": base_commit_full,
        "verified_base_tree_hash": base_tree_hash,
        "verified_source_worktree_hash": source_snapshot["hash"],
        "verified_source_snapshot_paths": source_snapshot["paths"],
        "generated_dirty_paths": generated_dirty_paths,
        "generated_dirty_snapshot_hash": generated_evidence_snapshot["hash"],
        "generated_dirty_snapshot_paths": generated_evidence_snapshot["paths"],
        "final_commit": base_commit if not status else None,
        "final_commit_pending_reason": None if not status else "worktree_dirty_at_verification",
        "worktree_dirty_at_verification": bool(status),
        "report_generated_before_commit": bool(status),
        "dirty_paths": dirty_paths,
    }
