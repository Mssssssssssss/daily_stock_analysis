# -*- coding: utf-8 -*-
"""
Market tools — wraps DataFetcherManager market-level methods as agent tools.

Tools:
- get_market_indices: major market index data
- get_sector_rankings: sector performance rankings
"""

import logging
from datetime import datetime, timezone

from src.agent.tools.registry import ToolParameter, ToolDefinition

logger = logging.getLogger(__name__)


def _get_fetcher_manager():
    """Lazy import to avoid circular deps."""
    from data_provider import DataFetcherManager
    return DataFetcherManager()


# ============================================================
# get_market_indices
# ============================================================

def _handle_get_market_indices(region: str = "cn") -> dict:
    """Get major market indices."""
    manager = _get_fetcher_manager()
    indices = manager.get_main_indices(region=region)

    fetched_at = datetime.now(timezone.utc).isoformat()
    if not indices:
        return {
            "error": f"No market index data available for region '{region}'",
            "fetched_at": fetched_at,
            "source": None,
            "provider_timestamp_missing": True,
        }

    return {
        "region": region,
        "indices_count": len(indices),
        "indices": indices,
        "fetched_at": fetched_at,
        "source": _extract_source(indices),
        "provider_timestamp_missing": not _has_provider_timestamp(indices),
    }


get_market_indices_tool = ToolDefinition(
    name="get_market_indices",
    description="Get major market indices (e.g., Shanghai Composite, Shenzhen Component, "
                "CSI 300 for China; S&P 500, Nasdaq, Dow for US). Provides market overview.",
    parameters=[
        ToolParameter(
            name="region",
            type="string",
            description="Market region: 'cn' for China A-shares, 'hk' for Hong Kong, 'us' for US stocks (default: 'cn')",
            required=False,
            default="cn",
            enum=["cn", "hk", "us"],
        ),
    ],
    handler=_handle_get_market_indices,
    category="market",
)


# ============================================================
# get_sector_rankings
# ============================================================

def _handle_get_sector_rankings(top_n: int = 10) -> dict:
    """Get sector performance rankings."""
    manager = _get_fetcher_manager()
    result = manager.get_sector_rankings(n=top_n)

    fetched_at = datetime.now(timezone.utc).isoformat()
    if result is None:
        return {
            "error": "No sector ranking data available",
            "fetched_at": fetched_at,
            "source": None,
            "provider_timestamp_missing": True,
        }

    # get_sector_rankings returns Tuple[List[Dict], List[Dict]]
    # (top_sectors, bottom_sectors)
    if isinstance(result, tuple) and len(result) == 2:
        top_sectors, bottom_sectors = result
        payload = {
            "top_sectors": top_sectors,
            "bottom_sectors": bottom_sectors,
        }
    elif isinstance(result, list):
        payload = {"sectors": result}
    else:
        payload = {"data": str(result)}
    payload.update({
        "fetched_at": fetched_at,
        "source": _extract_source(result),
        "provider_timestamp_missing": not _has_provider_timestamp(result),
    })
    return payload


def _extract_source(payload):
    if isinstance(payload, dict):
        return payload.get("source")
    if isinstance(payload, (list, tuple)):
        for item in payload:
            source = _extract_source(item)
            if source:
                return source
    return None


def _has_provider_timestamp(payload) -> bool:
    if isinstance(payload, dict):
        return any(payload.get(key) for key in ("provider_timestamp", "timestamp", "time", "datetime"))
    if isinstance(payload, (list, tuple)):
        return any(_has_provider_timestamp(item) for item in payload)
    return False


get_sector_rankings_tool = ToolDefinition(
    name="get_sector_rankings",
    description="Get sector/industry performance rankings. Returns top N and bottom N "
                "sectors by daily change percentage. Useful for sector rotation analysis.",
    parameters=[
        ToolParameter(
            name="top_n",
            type="integer",
            description="Number of top/bottom sectors to return (default: 10)",
            required=False,
            default=10,
        ),
    ],
    handler=_handle_get_sector_rankings,
    category="market",
)


ALL_MARKET_TOOLS = [
    get_market_indices_tool,
    get_sector_rankings_tool,
]
