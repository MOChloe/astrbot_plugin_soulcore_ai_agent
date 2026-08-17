"""Stable per-run limits for web and image research."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...contracts.web import WebSearchDepth, WebSearchIntensity


@dataclass(frozen=True, slots=True)
class WebIntensityLimits:
    provider_limits: Mapping[WebSearchDepth, int]
    searches_per_run: int
    reads_per_run: int
    result_limit: int
    data_budget_tokens: int


IMAGE_RESULT_LIMITS: Mapping[WebSearchIntensity, int] = {
    WebSearchIntensity.ECONOMY: 4,
    WebSearchIntensity.STANDARD: 6,
    WebSearchIntensity.DEEP: 8,
}


WEB_INTENSITY_LIMITS: Mapping[WebSearchIntensity, WebIntensityLimits] = {
    WebSearchIntensity.ECONOMY: WebIntensityLimits(
        {WebSearchDepth.QUICK: 1, WebSearchDepth.BALANCED: 1, WebSearchDepth.DEEP: 1},
        2,
        2,
        6,
        3500,
    ),
    WebSearchIntensity.STANDARD: WebIntensityLimits(
        {WebSearchDepth.QUICK: 1, WebSearchDepth.BALANCED: 2, WebSearchDepth.DEEP: 3},
        3,
        5,
        8,
        6000,
    ),
    WebSearchIntensity.DEEP: WebIntensityLimits(
        {WebSearchDepth.QUICK: 2, WebSearchDepth.BALANCED: 3, WebSearchDepth.DEEP: 4},
        5,
        8,
        12,
        10000,
    ),
}


__all__ = ["IMAGE_RESULT_LIMITS", "WEB_INTENSITY_LIMITS", "WebIntensityLimits"]
