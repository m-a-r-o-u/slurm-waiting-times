from datetime import datetime, timedelta, timezone

from slurm_waiting_times.models import SacctRow
from slurm_waiting_times.processing import filter_rows


TZ = timezone.utc


def make_row(job_id: str, submit_offset: int, start_offset: int, *, user: str = "alice", partition: str = "gpu") -> SacctRow:
    submit = datetime(2024, 5, 1, 12, 0, tzinfo=TZ) + timedelta(minutes=submit_offset)
    start = datetime(2024, 5, 1, 12, 0, tzinfo=TZ) + timedelta(minutes=start_offset)
    return SacctRow(
        job_id=job_id,
        user=user,
        submit_time=submit,
        start_time=start,
        state="COMPLETED",
        partition=partition,
    )


def test_filter_rows_excludes_steps_and_filters_users_partitions():
    rows = [
        make_row("123", 0, 10, user="alice", partition="gpu-a"),
        make_row("123.batch", 0, 10, user="alice", partition="gpu-a"),
        make_row("456", 0, 5, user="bob", partition="gpu-b"),
        make_row("789", 0, 15, user="carol", partition="cpu"),
    ]

    filtered = filter_rows(
        rows,
        include_steps=False,
        user_filters=["alice", "bob"],
        partition_filters=["gpu*"],
    )

    assert [record.job_id for record in filtered] == ["123", "456"]
    assert filtered[0].wait_seconds == 600
    assert filtered[1].wait_seconds == 300


def test_filter_rows_max_wait_hours():
    rows = [
        make_row("123", 0, 60),
        make_row("456", 0, 6 * 60),
    ]

    filtered = filter_rows(rows, max_wait_hours=1)
    assert [record.job_id for record in filtered] == ["123"]
