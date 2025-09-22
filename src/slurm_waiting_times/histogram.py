from __future__ import annotations

from statistics import mean
from typing import Sequence

try:  # pragma: no cover - dependency availability is environment specific
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
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
    ax.hist(values, bins=bins, edgecolor="black", color="#4C72B0")

    mean_seconds = mean(record.wait_seconds for record in records)
    mean_display = format_timedelta_hms(mean_seconds)
    if use_seconds:
        mean_line = mean_seconds
        xlabel = "Waiting time (seconds)"
    else:
        mean_line = mean_seconds / 60.0
        xlabel = "Waiting time (minutes)"

    ax.axvline(mean_line, color="red", linestyle="--", linewidth=2, label=f"Mean wait: {mean_display}")
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Job count")
    if title:
        ax.set_title(title)

    return fig
