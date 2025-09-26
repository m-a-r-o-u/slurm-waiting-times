from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("matplotlib")
from matplotlib import pyplot as plt

from slurm_waiting_times.histogram import (
    _format_time_value,
    create_histogram,
    prepare_histogram_values,
)
from slurm_waiting_times.models import JobRecord


BASE = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)


def make_record(wait_minutes: float) -> JobRecord:
    submit = BASE
    start = BASE + timedelta(minutes=wait_minutes)
    wait_seconds = (start - submit).total_seconds()
    return JobRecord(
        job_id="1",
        user="alice",
        submit_time=submit,
        start_time=start,
        state="COMPLETED",
        partition="gpu",
        wait_seconds=wait_seconds,
    )


def test_prepare_histogram_values_minutes():
    records = [make_record(5), make_record(10)]
    values = prepare_histogram_values(records, use_seconds=False)
    assert values == [5.0, 10.0]


def test_create_histogram_draws_mean_line():
    records = [make_record(5), make_record(15)]
    fig = create_histogram(records, use_seconds=False, bins=2, title="Example")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Waiting time [minutes]"
    assert ax.lines, "Expected a mean line"
    line = ax.lines[0]
    assert pytest.approx(line.get_xdata()[0], rel=1e-6) == 10.0
    assert "Mean wait" in line.get_label()
    fig.clf()
    plt.close(fig)


def test_format_time_value_rounds_to_minutes():
    # 89 seconds should round down, while 90 rounds up
    assert _format_time_value(89) == "0-00:01"
    assert _format_time_value(90) == "0-00:02"


def test_format_time_value_includes_days():
    total_seconds = (1 * 24 * 60 + 2 * 60 + 30) * 60
    assert _format_time_value(total_seconds) == "1-02:30"
