from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .analytics import analyze_history, compute_good_snapshots, compute_inequality_snapshot, compute_role_snapshots, summarize_recent_trends
from .config import SimulationConfig
from .dto import GoodRoleSnapshot, GoodSnapshot, InequalitySnapshot, MarketSnapshot, PhenomenaSnapshot, ProgressSnapshot, RecentTrendWindowSnapshot
from .long_run import load_checkpoint
from .metrics import MetricsSnapshot
from .service import SimulationService

_EPSILON = 1e-6


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
            status["read_only"] = False
            status["completed"] = False
            status["artifact_dir"] = None
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

    def get_recent_trends_payload(self, windows: tuple[int, ...] = (50, 100, 200)) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(item) for item in self.service.get_recent_trends_snapshot(windows=windows)]

    def get_role_mix_payload(self, *, limit: int = 10, sort_by: str = 'retailer_count') -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(item) for item in self.service.get_role_mix_snapshot(limit=limit, sort_by=sort_by)]

    def get_inequality_payload(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self.service.get_inequality_snapshot())

    def get_progress_payload(self) -> dict[str, Any]:
        with self._lock:
            cycle = self.service.engine.state.cycle
            return asdict(
                ProgressSnapshot(
                    current_cycle=cycle,
                    target_cycle=None,
                    progress_share=None,
                    checkpoint_updated_at=None,
                    checkpoint_age_seconds=None,
                    runner_log_updated_at=None,
                    runner_log_age_seconds=None,
                    latest_chunk_from_cycle=None,
                    latest_chunk_target_cycle=None,
                    recent_seconds_per_cycle=None,
                    eta_seconds=None,
                )
            )

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


def _empty_market_snapshot() -> MarketSnapshot:
    return MarketSnapshot.from_metrics(
        MetricsSnapshot(
            cycle=0,
            fulfilled_share=0.0,
            fulfilled_need_total=0.0,
            utility_proxy_total=0.0,
            unmet_need_total=0.0,
            stock_total=0.0,
            average_efficiency=0.0,
            mean_time_remaining=0.0,
            proposed_trade_count=0,
            accepted_trade_count=0,
            accepted_trade_volume=0.0,
            production_total=0.0,
            surplus_output_total=0.0,
            stock_consumption_total=0.0,
            leisure_extra_need_total=0.0,
            inventory_trade_volume=0.0,
            network_density=0.0,
            monetary_concentration=0.0,
            rare_goods_monetary_share=0.0,
            average_needs_level=0.0,
            periodic_tce_cost_total=0.0,
            periodic_spoilage_total=0.0,
            tce_cost_in_time_total=0.0,
            spoilage_cost_in_time_total=0.0,
            stored_delta_total=0.0,
            loser_share=0.0,
            price_average=0.0,
        )
    )


def _empty_inequality_snapshot() -> InequalitySnapshot:
    return InequalitySnapshot(
        stock_value_gini=0.0,
        stock_value_top_decile_share=0.0,
        stock_value_mean=0.0,
        stock_value_median=0.0,
        living_standard_gini=0.0,
        living_standard_top_decile_share=0.0,
        living_standard_mean=0.0,
        living_standard_median=0.0,
        living_standard_p10=0.0,
        living_standard_p25=0.0,
        living_standard_p75=0.0,
        living_standard_p90=0.0,
        living_standard_p99=0.0,
        aspiration_balance_mean=0.0,
        aspiration_balance_median=0.0,
        aspiration_balance_p10=0.0,
        aspiration_balance_p90=0.0,
        aspiration_shortfall_share=0.0,
        aspiration_shortfall_mean=0.0,
        aspiration_shortfall_p90=0.0,
        smith_cost_gini=0.0,
        smith_cost_top_decile_share=0.0,
        smith_cost_mean=0.0,
        smith_cost_median=0.0,
        smith_cost_p10=0.0,
        smith_cost_p25=0.0,
        smith_cost_p75=0.0,
        smith_cost_p90=0.0,
        smith_cost_p99=0.0,
        production_time_value=0.0,
        direct_production_time=0.0,
        production_time_share_of_budget=0.0,
        tce_share_of_output_value=0.0,
        spoilage_share_of_output_value=0.0,
        friction_share_of_output_value=0.0,
        tce_share_of_time_budget=0.0,
        spoilage_share_of_time_budget=0.0,
        friction_share_of_time_budget=0.0,
    )


def _market_snapshot_from_payload(payload: dict[str, Any]) -> MarketSnapshot:
    return MarketSnapshot.from_metrics(MetricsSnapshot(**payload))


class ArtifactDashboardController:
    def __init__(self, artifact_dir: str | Path, *, top_goods: int = 10) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.top_goods = top_goods
        self._lock = threading.Lock()
        self._last_error: str | None = None
        self._metrics_signature: tuple[bool, int, int] | None = None
        self._checkpoint_signature: tuple[bool, int, int, int, int] | None = None
        self._summary_signature: tuple[bool, int, int] | None = None
        self._metrics_history: list[MarketSnapshot] = []
        self._checkpoint_history: list[MarketSnapshot] = []
        self._current_market = _empty_market_snapshot()
        self._goods: list[GoodSnapshot] = []
        self._role_goods: list[GoodRoleSnapshot] = []
        self._inequality = _empty_inequality_snapshot()
        self._progress = ProgressSnapshot(
            current_cycle=0,
            target_cycle=None,
            progress_share=None,
            checkpoint_updated_at=None,
            checkpoint_age_seconds=None,
            runner_log_updated_at=None,
            runner_log_age_seconds=None,
            latest_chunk_from_cycle=None,
            latest_chunk_target_cycle=None,
            recent_seconds_per_cycle=None,
            eta_seconds=None,
        )
        self._phenomena = analyze_history([], [])
        self._config: dict[str, Any] | None = None
        self._backend_name = "artifact"
        self._device = "artifact observer"
        self._completed = False
        self._refresh_locked()

    def close(self) -> None:
        return None

    def get_status_payload(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            history_length = len(self._metrics_history) if self._metrics_history else len(self._checkpoint_history)
            return {
                "cycle": self._current_market.cycle,
                "backend_name": self._backend_name,
                "device": self._device,
                "history_length": history_length,
                "is_running": (not self._completed) and self._current_market.cycle > 0,
                "config": self._config,
                "batch_cycles": None,
                "last_error": self._last_error,
                "read_only": True,
                "completed": self._completed,
                "artifact_dir": str(self.artifact_dir.resolve()),
            }

    def get_market_payload(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            return asdict(self._current_market)

    def get_history_payload(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_locked()
            history = self._metrics_history or self._checkpoint_history
            if limit is not None:
                history = history[-limit:]
            if history and history[-1].cycle == self._current_market.cycle:
                history = [*history[:-1], self._current_market]
            return [asdict(item) for item in history]

    def get_goods_payload(self, *, limit: int = 10, sort_by: str = "monetary_score") -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        valid_sort_keys = {
            "monetary_score",
            "value_weighted_monetary_score",
            "recent_purchase_total",
            "recent_purchase_value_total",
            "recent_inventory_inflow_total",
            "recent_inventory_inflow_value_total",
            "stock_total",
            "average_efficiency",
            "base_need",
            "exchange_media_score",
            "relative_tce_loss",
            "relative_trade_flow",
            "relative_stock",
            "network_circulation_breadth",
            "excess_stock_breadth",
            "excess_stock_ratio",
            "round_trip_breadth",
            "round_trip_turnover_share",
            "consumer_flow_share",
            "retailer_stock_share",
            "local_liquidity_score",
            "local_liquidity_acceptance_breadth",
            "local_liquidity_visible_acceptance",
            "local_liquidity_target_increment",
            "exchange_media_reserve_score",
            "exchange_media_reserve_scale",
            "exchange_media_reserve_gap",
            "exchange_media_spread_ok_share",
        }
        if sort_by not in valid_sort_keys:
            raise ValueError(f"Unsupported sort field: {sort_by}")
        with self._lock:
            self._refresh_locked()
            goods = sorted(self._goods, key=lambda item: (getattr(item, sort_by), -item.good_id), reverse=True)[:limit]
            return [asdict(item) for item in goods]

    def get_phenomena_payload(self, *, top_goods: int = 10) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            if top_goods <= 0:
                raise ValueError("top_goods must be positive")
            history = self._checkpoint_history or self._metrics_history
            goods = self._goods[:top_goods]
            if history and goods:
                self._phenomena = analyze_history(history, goods)
            return asdict(self._phenomena)

    def get_recent_trends_payload(self, windows: tuple[int, ...] = (50, 100, 200)) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_locked()
            history = self._checkpoint_history or self._metrics_history
            return [asdict(item) for item in summarize_recent_trends(history, windows=windows)]

    def get_role_mix_payload(self, *, limit: int = 10, sort_by: str = 'retailer_count') -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_locked()
            if limit <= 0:
                raise ValueError('limit must be positive')
            valid_sort_keys = {
                'retailer_count',
                'producer_count',
                'consumer_count',
                'retailer_inventory_inflow_total',
                'retailer_sales_total',
                'retailer_purchase_total',
                'base_need',
            }
            if sort_by not in valid_sort_keys:
                raise ValueError(f'Unsupported sort field: {sort_by}')
            goods = sorted(self._role_goods, key=lambda item: (getattr(item, sort_by), -item.good_id), reverse=True)[:limit]
            return [asdict(item) for item in goods]

    def get_inequality_payload(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            return asdict(self._inequality)

    def get_progress_payload(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            return asdict(self._progress)

    def reset(self, payload: dict[str, Any]) -> None:
        raise RuntimeError("Dashboard is observing run artifacts in read-only mode")

    def start(self) -> None:
        raise RuntimeError("Dashboard is observing run artifacts in read-only mode")

    def pause(self) -> None:
        raise RuntimeError("Dashboard is observing run artifacts in read-only mode")

    def step(self, cycles: int) -> None:
        raise RuntimeError("Dashboard is observing run artifacts in read-only mode")

    def _refresh_locked(self) -> None:
        try:
            self._refresh_metrics_locked()
            self._refresh_checkpoint_locked()
            self._refresh_summary_locked()
            self._update_current_market_locked()
            self._refresh_progress_locked()
            if self._checkpoint_history and self._goods:
                self._phenomena = analyze_history(self._checkpoint_history, self._goods[: self.top_goods])
            elif self._metrics_history and self._goods:
                self._phenomena = analyze_history(self._metrics_history, self._goods[: self.top_goods])
            self._last_error = None
        except Exception as exc:  # pragma: no cover - observer runtime path
            self._last_error = str(exc)

    def _refresh_metrics_locked(self) -> None:
        metrics_path = self.artifact_dir / "metrics.jsonl"
        signature = (
            metrics_path.exists(),
            int(metrics_path.stat().st_mtime_ns) if metrics_path.exists() else 0,
            int(metrics_path.stat().st_size) if metrics_path.exists() else 0,
        )
        if signature == self._metrics_signature:
            return
        self._metrics_signature = signature
        if not metrics_path.exists():
            self._metrics_history = []
            return
        snapshots: list[MarketSnapshot] = []
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                snapshots.append(_market_snapshot_from_payload(json.loads(line)))
        self._metrics_history = snapshots

    def _refresh_checkpoint_locked(self) -> None:
        checkpoint_json = self.artifact_dir / "checkpoint_latest.json"
        checkpoint_npz = self.artifact_dir / "checkpoint_latest.npz"
        signature = (
            checkpoint_json.exists(),
            int(checkpoint_json.stat().st_mtime_ns) if checkpoint_json.exists() else 0,
            int(checkpoint_json.stat().st_size) if checkpoint_json.exists() else 0,
            int(checkpoint_npz.stat().st_mtime_ns) if checkpoint_npz.exists() else 0,
            int(checkpoint_npz.stat().st_size) if checkpoint_npz.exists() else 0,
        )
        if signature == self._checkpoint_signature:
            return
        self._checkpoint_signature = signature
        if not checkpoint_json.exists():
            self._checkpoint_history = []
            self._goods = []
            self._role_goods = []
            self._inequality = _empty_inequality_snapshot()
            return

        metadata = json.loads(checkpoint_json.read_text(encoding="utf-8"))
        self._config = metadata.get("config")
        self._backend_name = str(metadata.get("backend_name", self._backend_name))
        self._checkpoint_history = [_market_snapshot_from_payload(item) for item in metadata.get("history", [])]

        if checkpoint_npz.exists():
            engine = load_checkpoint(self.artifact_dir)
            self._goods = compute_good_snapshots(
                state=engine.state,
                backend=engine.backend,
                config=engine.config,
                limit=None,
            )
            self._role_goods = compute_role_snapshots(
                state=engine.state,
                backend=engine.backend,
                limit=None,
            )
            self._inequality = compute_inequality_snapshot(
                state=engine.state,
                backend=engine.backend,
                config=engine.config,
            )
            exchange_media_concentration, rare_goods_exchange_media_share = _exchange_media_aggregate_from_goods(self._goods)
            if self._checkpoint_history:
                self._checkpoint_history[-1] = replace(
                    self._checkpoint_history[-1],
                    exchange_media_concentration=exchange_media_concentration,
                    rare_goods_exchange_media_share=rare_goods_exchange_media_share,
                )
            if self._metrics_history and self._metrics_history[-1].cycle == engine.cycle:
                self._metrics_history[-1] = replace(
                    self._metrics_history[-1],
                    exchange_media_concentration=exchange_media_concentration,
                    rare_goods_exchange_media_share=rare_goods_exchange_media_share,
                )
            if self._current_market.cycle <= engine.cycle:
                self._current_market = replace(
                    self._current_market,
                    exchange_media_concentration=exchange_media_concentration,
                    rare_goods_exchange_media_share=rare_goods_exchange_media_share,
                )

    def _refresh_summary_locked(self) -> None:
        summary_path = self.artifact_dir / "summary.json"
        signature = (
            summary_path.exists(),
            int(summary_path.stat().st_mtime_ns) if summary_path.exists() else 0,
            int(summary_path.stat().st_size) if summary_path.exists() else 0,
        )
        if signature == self._summary_signature:
            return
        self._summary_signature = signature
        self._completed = summary_path.exists()
        if not summary_path.exists():
            return
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self._config = summary.get("config", self._config)
        self._backend_name = str(summary.get("backend_name", self._backend_name))
        latest_market = summary.get("latest_market")
        if latest_market is not None and int(latest_market.get("cycle", -1)) >= self._current_market.cycle:
            self._current_market = MarketSnapshot(**latest_market)
        phenomena = summary.get("phenomena")
        if phenomena is not None:
            self._phenomena = PhenomenaSnapshot(**phenomena)

    def _update_current_market_locked(self) -> None:
        candidates = [self._current_market]
        if self._checkpoint_history:
            candidates.append(self._checkpoint_history[-1])
        if self._metrics_history:
            candidates.append(self._metrics_history[-1])
        self._current_market = max(enumerate(candidates), key=lambda item: (item[1].cycle, item[0]))[1]

    def _refresh_progress_locked(self) -> None:
        checkpoint_path = self.artifact_dir / "checkpoint_latest.json"
        runner_log_path = self.artifact_dir / "runner.log"
        extension_log_path = self.artifact_dir / "extension_runner.log"
        checkpoint_updated_at = _format_timestamp(_mtime_or_none(checkpoint_path))
        checkpoint_age_seconds = _age_seconds(_mtime_or_none(checkpoint_path))
        runner_log_updated_at = _format_timestamp(_mtime_or_none(runner_log_path))
        runner_log_age_seconds = _age_seconds(_mtime_or_none(runner_log_path))

        target_cycle: int | None = None
        latest_chunk_from_cycle: int | None = None
        latest_chunk_target_cycle: int | None = None
        recent_seconds_per_cycle: float | None = None

        if runner_log_path.exists():
            text = runner_log_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
            if extension_log_path.exists():
                text += "\n" + extension_log_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
            target_matches = re.findall(r"target_cycle=(\d+)", text)
            if target_matches:
                target_cycle = max(int(value) for value in target_matches)

            chunk_matches = re.findall(r"starting chunk from_cycle=(\d+) chunk_cycles=(\d+)", text)
            if chunk_matches:
                latest_from, latest_chunk_cycles = chunk_matches[-1]
                latest_chunk_from_cycle = int(latest_from)
                latest_chunk_target_cycle = int(latest_from) + int(latest_chunk_cycles)

            speed_matches = re.findall(r"long_run start=(\d+) end=(\d+) seconds=([0-9.]+)", text)
            speeds = []
            for start_value, end_value, seconds_value in speed_matches[-5:]:
                delta_cycles = max(1, int(end_value) - int(start_value))
                speeds.append(float(seconds_value) / delta_cycles)
            if speeds:
                recent_seconds_per_cycle = sum(speeds) / len(speeds)

        progress_share: float | None = None
        eta_seconds: float | None = None
        if target_cycle and target_cycle > 0:
            progress_share = min(1.0, max(0.0, self._current_market.cycle / target_cycle))
            if recent_seconds_per_cycle is not None and self._current_market.cycle < target_cycle:
                eta_seconds = (target_cycle - self._current_market.cycle) * recent_seconds_per_cycle
            # Chunked long runs rewrite summary.json after every chunk. Treat the
            # run as completed only after the runner target is actually reached.
            self._completed = self._current_market.cycle >= target_cycle

        self._progress = ProgressSnapshot(
            current_cycle=self._current_market.cycle,
            target_cycle=target_cycle,
            progress_share=progress_share,
            checkpoint_updated_at=checkpoint_updated_at,
            checkpoint_age_seconds=checkpoint_age_seconds,
            runner_log_updated_at=runner_log_updated_at,
            runner_log_age_seconds=runner_log_age_seconds,
            latest_chunk_from_cycle=latest_chunk_from_cycle,
            latest_chunk_target_cycle=latest_chunk_target_cycle,
            recent_seconds_per_cycle=recent_seconds_per_cycle,
            eta_seconds=eta_seconds,
        )


class _DashboardHandler(BaseHTTPRequestHandler):
    controller: Any
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
        if parsed.path == "/api/recent_trends":
            self._send_json(self.controller.get_recent_trends_payload())
            return
        if parsed.path == "/api/roles":
            limit = _optional_int(params.get("limit", [10])[0]) or 10
            sort_by = params.get("sort_by", ["retailer_count"])[0]
            self._send_json(self.controller.get_role_mix_payload(limit=limit, sort_by=sort_by))
            return
        if parsed.path == "/api/inequality":
            self._send_json(self.controller.get_inequality_payload())
            return
        if parsed.path == "/api/progress":
            self._send_json(self.controller.get_progress_payload())
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


def _mtime_or_none(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_mtime


def _format_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _age_seconds(timestamp: float | None) -> float | None:
    if timestamp is None:
        return None
    return max(0.0, time.time() - timestamp)


def _exchange_media_aggregate_from_goods(goods: list[GoodSnapshot]) -> tuple[float, float]:
    total_score = sum(max(item.exchange_media_score, 0.0) for item in goods)
    if total_score <= _EPSILON:
        return 0.0, 0.0
    concentration = sum((max(item.exchange_media_score, 0.0) / total_score) ** 2 for item in goods)
    rare_share = sum(max(item.exchange_media_score, 0.0) for item in goods if item.is_rare) / total_score
    return float(concentration), float(rare_share)


def serve_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    config: SimulationConfig | None = None,
    backend_name: str = "numpy",
    batch_cycles: int = 1,
    artifact_dir: str | Path | None = None,
) -> None:
    html = Path(__file__).with_name("dashboard_page.html").read_bytes()
    controller: Any
    if artifact_dir is None:
        controller = DashboardController(
            SimulationService.create(config=config, backend_name=backend_name),
            batch_cycles=batch_cycles,
        )
    else:
        controller = ArtifactDashboardController(artifact_dir)

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
