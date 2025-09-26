from __future__ import annotations

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

    fig, ax = plt.subplots(figsize=(10, 6))

    # Matplotlib cannot render logarithmic axes that include non-positive values.
    # Replace zeros with a small positive value to keep them visible on the log
    # scale without altering their bin membership in a meaningful way.
    min_positive = min((value for value in values if value > 0), default=None)
    if min_positive is None:
        # All waiting times are zero – fall back to plotting a single bin.
        min_positive = 1e-3
    adjusted_values = [value if value > 0 else min_positive / 2 for value in values]

    ax.hist(adjusted_values, bins=bins, edgecolor="black", color="#4C72B0")

    wait_seconds = [record.wait_seconds for record in records]
    mean_seconds = mean(wait_seconds)
    median_seconds = median(wait_seconds)
    mean_display = format_timedelta_hms(mean_seconds)
    if use_seconds:
        mean_line = mean_seconds
        median_line = median_seconds
        xlabel = "Waiting time [seconds]"
        tick_formatter = None
    else:
        mean_line = mean_seconds / 60.0
        median_line = median_seconds / 60.0
        xlabel = "Waiting time [minutes]"

        def tick_formatter(_: float, pos: int) -> str:  # pragma: no cover - simple formatting
            del pos
            value = _
            if value < 1:
                minutes = value
                seconds = minutes * 60
                if seconds < 1:
                    return f"{seconds*1000:.0f} ms"
                return f"{seconds:.0f} s"
            if value < 60:
                return f"{value:.0f} min"
            if value < 1440:
                hours = value / 60
                if hours.is_integer():
                    return f"{hours:.0f} h"
                return f"{hours:.1f} h"
            days = value / 1440
            if days.is_integer():
                return f"{days:.0f} d"
            return f"{days:.1f} d"

    if mean_line <= 0:
        mean_line = min_positive / 2

    ax.axvline(
        mean_line,
        color="#009FE3",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean wait: {mean_display}",
    )

    if median_seconds > 0:
        median_display = format_timedelta_hms(median_seconds)
        ax.axvline(
            median_line,
            color="#6E6E6E",
            linestyle="--",
            linewidth=1.5,
            label=f"Median wait: {median_display}",
        )

    ax.set_xscale("log")
    if tick_formatter is not None:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(tick_formatter))
        candidate_ticks = [
            0.1,
            0.5,
            1,
            5,
            10,
            30,
            60,
            120,
            300,
            600,
            1440,
            2880,
            7200,
        ]
        x_min, x_max = min(adjusted_values), max(adjusted_values)
        ticks = [tick for tick in candidate_ticks if x_min <= tick <= x_max]
        if ticks:
            ax.set_xticks(ticks)

    ax.legend(fontsize=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Job count", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    if title:
        ax.set_title(title, fontsize=16)

    ax.yaxis.grid(True, color="#D3D3D3", linestyle="--", linewidth=0.7, alpha=0.7)

    ymin, ymax = ax.get_ylim()
    text_y = ymin + 0.02 * (ymax - ymin)
    ax.annotate(
        f"Mean: {mean_display}",
        xy=(mean_line, text_y),
        xytext=(5, 5),
        textcoords="offset points",
        color="#009FE3",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.tight_layout()

    return fig
