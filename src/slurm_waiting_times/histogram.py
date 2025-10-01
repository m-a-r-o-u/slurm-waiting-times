from __future__ import annotations

import math
from statistics import mean, median
from typing import Sequence

try:  # pragma: no cover - dependency availability is environment specific
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, ticker
except Exception:  # pragma: no cover - handled at runtime
    matplotlib = None
    plt = None

from .models import JobRecord
from .time_utils import format_timedelta_hms, freedman_diaconis_bins


def prepare_histogram_values(records: Sequence[JobRecord], *, use_seconds: bool) -> list[float]:
    if use_seconds:
        return [record.wait_seconds for record in records]
    return [record.wait_seconds / 60.0 for record in records]


LRZ_SKY_BLUE = "#009FE3"

_NICE_TIME_SECONDS = [
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    15,
    30,
    45,
    60,
    120,
    300,
    600,
    900,
    1800,
    2700,
    3600,
    5400,
    7200,
    10800,
    14400,
    21600,
    28800,
    43200,
    64800,
    86400,
    172800,
    345600,
    604800,
]


def _format_time_value(total_seconds: float) -> str:
    return format_timedelta_hms(total_seconds)


def _nice_ticks(min_value: float, max_value: float, *, use_seconds: bool) -> list[float]:
    if min_value <= 0:
        min_value = min(tick for tick in _NICE_TIME_SECONDS if tick > 0)

    min_seconds = min_value if use_seconds else min_value * 60
    max_seconds = max_value if use_seconds else max_value * 60
    if min_seconds == max_seconds:
        return [min_value]

    ticks = [tick for tick in _NICE_TIME_SECONDS if min_seconds <= tick <= max_seconds]
    if not ticks:
        ticks = [min_seconds, max_seconds]
    else:
        max_ticks = 6
        if len(ticks) > max_ticks:
            step = math.ceil(len(ticks) / max_ticks)
            ticks = ticks[::step]
    if use_seconds:
        return ticks
    return [tick / 60 for tick in ticks]


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between 0 and 1")
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _logspace_bins(data: Sequence[float], *, bin_count: int) -> list[float]:
    if bin_count < 1:
        raise ValueError("bin_count must be at least 1")
    positive = [value for value in data if value > 0]
    if not positive:
        positive = [1e-3]
    start = min(positive)
    end = max(positive)
    if start == end:
        return [start * 0.8, end * 1.2]
    log_start = math.log10(start)
    log_end = math.log10(end)
    step = (log_end - log_start) / bin_count
    return [10 ** (log_start + i * step) for i in range(bin_count + 1)]


def create_histogram(
    records: Sequence[JobRecord],
    *,
    use_seconds: bool = False,
    bins: int | None = None,
    title: str = "",
) -> plt.Figure:
    if not records:
        raise ValueError("create_histogram requires at least one record")

    if plt is None:
        raise RuntimeError("matplotlib is required to create histograms")

    values = prepare_histogram_values(records, use_seconds=use_seconds)

    if bins is None:
        bins = freedman_diaconis_bins(values)

    fig, (ax_typical, ax_tail) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Matplotlib cannot render logarithmic axes that include non-positive values.
    # Replace zeros with a small positive value to keep them visible on the log
    # scale without altering their bin membership in a meaningful way.
    min_positive = min((value for value in values if value > 0), default=None)
    if min_positive is None:
        # All waiting times are zero – fall back to plotting a single bin.
        min_positive = 1e-3
    adjusted_values = [value if value > 0 else min_positive / 2 for value in values]

    if bins is None:
        bins = freedman_diaconis_bins(adjusted_values)

    bins = max(1, min(80, bins))

    sorted_values = sorted(adjusted_values)
    typical_cutoff = _percentile(sorted_values, 0.95)
    typical_values = [value for value in adjusted_values if value <= typical_cutoff]
    tail_values = [value for value in adjusted_values if value > typical_cutoff]

    typical_bins = _logspace_bins(typical_values or adjusted_values, bin_count=bins)
    ax_typical.hist(
        typical_values or adjusted_values,
        bins=typical_bins,
        color=LRZ_SKY_BLUE,
        alpha=0.75,
        edgecolor="none",
    )

    if tail_values:
        tail_bins = _logspace_bins(tail_values, bin_count=max(1, bins // 2))
        ax_tail.hist(
            tail_values,
            bins=tail_bins,
            color=LRZ_SKY_BLUE,
            alpha=0.75,
            edgecolor="none",
        )
    else:
        ax_tail.set_axis_off()

    wait_seconds = [record.wait_seconds for record in records]
    mean_seconds = mean(wait_seconds)
    median_seconds = median(wait_seconds)
    mean_display = format_timedelta_hms(mean_seconds)
    if use_seconds:
        xlabel = "Waiting time [seconds]"
        mean_line_value = mean_seconds
    else:
        xlabel = "Waiting time"
        mean_line_value = mean_seconds / 60.0

    def tick_formatter(value: float, pos: int) -> str:  # pragma: no cover - simple formatting
        del pos
        total_seconds = value if use_seconds else value * 60
        return _format_time_value(total_seconds)

    if mean_line_value <= 0:
        mean_line_value = min_positive / 2

    ax_typical.axvline(
        mean_line_value,
        color=LRZ_SKY_BLUE,
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label=f"Mean wait: {mean_display}",
    )

    axes = [
        (ax_typical, typical_values or adjusted_values),
        (ax_tail, tail_values),
    ]
    formatter = ticker.FuncFormatter(tick_formatter)
    for ax, axis_values in axes:
        if not ax.has_data() or not axis_values:
            continue
        ax.set_xscale("log")
        ticks = _nice_ticks(min(axis_values), max(axis_values), use_seconds=use_seconds)
        if ticks:
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.xaxis.get_offset_text().set_visible(False)
        ax.tick_params(axis="x", labelsize=11, colors="#303030", rotation=25)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.tick_params(axis="y", labelsize=11, colors="#303030")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.grid(True, color="#B7D9F2", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_facecolor("white")
        ax.margins(x=0.02)

    ax_typical.set_xlabel(xlabel, fontsize=13, color="#202020")
    if tail_values:
        ax_tail.tick_params(labelleft=False)
        ax_tail.set_xlabel(xlabel, fontsize=13, color="#202020")

    ax_typical.set_ylabel("Job count", fontsize=13, color="#202020")

    if title:
        fig.suptitle(title, fontsize=16, color="#202020", y=0.98)

    ax_typical.set_title("Typical waits (≤95th percentile)", fontsize=12, color="#202020")
    if tail_values:
        ax_tail.set_title("Long tail (>95th percentile)", fontsize=12, color="#202020")

    p95_seconds = _percentile(sorted(wait_seconds), 0.95)
    max_seconds = max(wait_seconds)
    stats_lines = [
        f"Jobs: {len(records)}",
        f"Mean: {mean_display}",
        f"Median: {format_timedelta_hms(median_seconds)}",
        f"95th: {format_timedelta_hms(p95_seconds)}",
        f"Max: {format_timedelta_hms(max_seconds)}",
    ]
    ax_typical.text(
        0.98,
        0.98,
        "\n".join(stats_lines),
        transform=ax_typical.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#E6F3FB",
            "edgecolor": LRZ_SKY_BLUE,
            "linewidth": 0.8,
        },
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    return fig
