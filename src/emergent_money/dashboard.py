from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import SimulationConfig
from .service import SimulationService


class DashboardController:
    def __init__(self, service: SimulationService, *, batch_cycles: int = 1, poll_interval: float = 0.05) -> None:
        self.service = service
        self.batch_cycles = batch_cycles
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._run_flag = threading.Event()
        self._stop_flag = threading.Event()
        self._last_error: str | None = None
        self._worker = threading.Thread(target=self._run_loop, daemon=True, name="emergent-money-dashboard")
        self._worker.start()

    def close(self) -> None:
        self._stop_flag.set()
        self._run_flag.clear()
        self._worker.join(timeout=1.0)

    def get_status_payload(self) -> dict[str, Any]:
        with self._lock:
            status = asdict(self.service.get_status())
            status["is_running"] = self._run_flag.is_set()
            status["config"] = asdict(self.service.engine.config)
            status["batch_cycles"] = self.batch_cycles
            status["last_error"] = self._last_error
            return status

    def get_market_payload(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self.service.get_market_snapshot())

    def get_history_payload(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(item) for item in self.service.get_history(limit=limit)]

    def get_goods_payload(self, *, limit: int = 10, sort_by: str = "monetary_score") -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(item) for item in self.service.get_goods_snapshot(limit=limit, sort_by=sort_by)]

    def get_phenomena_payload(self, *, top_goods: int = 10) -> dict[str, Any]:
        with self._lock:
            return asdict(self.service.get_phenomena_snapshot(top_goods=top_goods))

    def reset(self, payload: dict[str, Any]) -> None:
        config_payload = payload.get("config", {})
        current = asdict(self.service.engine.config)
        current.update(config_payload)
        backend_name = payload.get("backend_name") or self.service.engine.backend.metadata.name
        batch_cycles = int(payload.get("batch_cycles", self.batch_cycles))
        if batch_cycles <= 0:
            raise ValueError("batch_cycles must be positive")
        config = SimulationConfig(**current)
        with self._lock:
            self.batch_cycles = batch_cycles
            self._last_error = None
            self._run_flag.clear()
            self.service.reset(config=config, backend_name=backend_name)

    def start(self) -> None:
        self._last_error = None
        self._run_flag.set()

    def pause(self) -> None:
        self._run_flag.clear()
        with self._lock:
            self.service.pause()

    def step(self, cycles: int) -> None:
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        self._run_flag.clear()
        with self._lock:
            self._last_error = None
            self.service.step(cycles)

    def _run_loop(self) -> None:
        while not self._stop_flag.is_set():
            if not self._run_flag.is_set():
                time.sleep(self.poll_interval)
                continue
            try:
                with self._lock:
                    self.service.resume()
                    self.service.step(self.batch_cycles)
            except Exception as exc:  # pragma: no cover - runtime UI path
                self._last_error = str(exc)
                self._run_flag.clear()
            time.sleep(self.poll_interval)


class _DashboardHandler(BaseHTTPRequestHandler):
    controller: DashboardController
    html: bytes

    def do_GET(self) -> None:  # pragma: no cover - manual UI path
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/":
            self._send_bytes(self.html, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self._send_json(self.controller.get_status_payload())
            return
        if parsed.path == "/api/market":
            self._send_json(self.controller.get_market_payload())
            return
        if parsed.path == "/api/history":
            self._send_json(self.controller.get_history_payload(limit=_optional_int(params.get("limit", [None])[0])))
            return
        if parsed.path == "/api/goods":
            limit = _optional_int(params.get("limit", [10])[0]) or 10
            sort_by = params.get("sort_by", ["monetary_score"])[0]
            self._send_json(self.controller.get_goods_payload(limit=limit, sort_by=sort_by))
            return
        if parsed.path == "/api/phenomena":
            top_goods = _optional_int(params.get("top_goods", [10])[0]) or 10
            self._send_json(self.controller.get_phenomena_payload(top_goods=top_goods))
            return
        self._send_text(HTTPStatus.NOT_FOUND, "Unknown route")

    def do_POST(self) -> None:  # pragma: no cover - manual UI path
        parsed = urlparse(self.path)
        payload = self._read_body()
        try:
            if parsed.path == "/api/reset":
                self.controller.reset(payload)
            elif parsed.path == "/api/start":
                self.controller.start()
            elif parsed.path == "/api/pause":
                self.controller.pause()
            elif parsed.path == "/api/step":
                self.controller.step(int(payload.get("cycles", 1)))
            else:
                self._send_text(HTTPStatus.NOT_FOUND, "Unknown route")
                return
        except Exception as exc:
            self._send_text(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_json({"ok": True})

    def log_message(self, format: str, *args) -> None:  # pragma: no cover
        return None

    def _read_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if not raw_length:
            return {}
        raw = self.rfile.read(int(raw_length))
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _send_json(self, payload: Any) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _send_text(self, status: HTTPStatus, message: str) -> None:
        data = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def serve_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    config: SimulationConfig | None = None,
    backend_name: str = "numpy",
    batch_cycles: int = 1,
) -> None:
    html = Path(__file__).with_name("dashboard_page.html").read_bytes()
    controller = DashboardController(
        SimulationService.create(config=config, backend_name=backend_name),
        batch_cycles=batch_cycles,
    )

    class Handler(_DashboardHandler):
        pass

    Handler.controller = controller
    Handler.html = html

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        controller.close()
        server.server_close()


def main() -> int:
    serve_dashboard()
    return 0
