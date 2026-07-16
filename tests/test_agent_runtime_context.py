# -*- coding: utf-8 -*-
"""Regression tests for server-owned Agent date semantics."""
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import patch

from src.core import trading_calendar


def test_resolve_agent_market_prefers_stock_code_and_market_words() -> None:
    assert trading_calendar.resolve_agent_market("看看港股", {"stock_code": "AAPL"}) == "us"
    assert trading_calendar.resolve_agent_market("恒生指数怎么样") == "hk"
    assert trading_calendar.resolve_agent_market("美股大盘") == "us"
    assert trading_calendar.resolve_agent_market("分析 AAPL") == "us"
    assert trading_calendar.resolve_agent_market("今天大盘") == "cn"


def test_runtime_context_does_not_reuse_client_time_fields() -> None:
    phase = SimpleNamespace(
        market_local_time=datetime(2026, 4, 24, 10, 0),
        session_date=date(2026, 4, 24),
        effective_daily_bar_date=date(2026, 4, 23),
        warnings=[],
        to_dict=lambda: {"market": "cn", "effective_daily_bar_date": "2026-04-23"},
    )
    with patch.object(trading_calendar, "_XCALS_AVAILABLE", True), patch.object(
        trading_calendar, "build_market_phase_context", return_value=phase,
    ):
        runtime = trading_calendar.build_agent_runtime_context(
            task="分析 600519",
            context={"stock_code": "600519", "market_phase_context": {"effective_daily_bar_date": "2000-01-01"}},
        )

    assert runtime.latest_complete_daily_bar_date == date(2026, 4, 23)
    assert runtime.to_dict()["market_natural_date"] == "2026-04-24"
    assert runtime.to_dict()["market_weekday"] == "星期五"


def test_runtime_context_exposes_server_owned_market_weekday() -> None:
    phase = SimpleNamespace(
        market_local_time=datetime(2026, 7, 16, 10, 0),
        session_date=date(2026, 7, 16),
        effective_daily_bar_date=date(2026, 7, 15),
        warnings=[],
        to_dict=lambda: {"market": "cn", "effective_daily_bar_date": "2026-07-15"},
    )
    with patch.object(trading_calendar, "_XCALS_AVAILABLE", True), patch.object(
        trading_calendar, "build_market_phase_context", return_value=phase,
    ):
        runtime = trading_calendar.build_agent_runtime_context(task="分析 600519")

    payload = runtime.to_dict()
    assert payload["market_natural_date"] == "2026-07-16"
    assert payload["market_weekday"] == "星期四"
    assert payload["effective_daily_bar_date"] == "2026-07-15"


def test_runtime_context_marks_unknown_daily_bar_without_calendar() -> None:
    with patch.object(trading_calendar, "_XCALS_AVAILABLE", False):
        runtime = trading_calendar.build_agent_runtime_context(
            task="分析 600519",
            current_time=datetime(2026, 4, 24, 10, 0),
        )

    assert runtime.latest_complete_daily_bar_date is None
    assert "calendar_unavailable" in runtime.market_phase_context.warnings
