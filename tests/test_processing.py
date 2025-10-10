from datetime import datetime, timedelta, timezone

from slurm_waiting_times.models import SacctRow
from slurm_waiting_times.processing import determine_job_type, filter_rows


TZ = timezone.utc


def make_row(
    job_id: str,
    submit_offset: int,
    start_offset: int,
    *,
    user: str = "alice",
    partition: str = "gpu",
    nodes: int | None = 1,
    alloc_gres: str | None = "gpu:1",
) -> SacctRow:
    submit = datetime(2024, 5, 1, 12, 0, tzinfo=TZ) + timedelta(minutes=submit_offset)
    start = datetime(2024, 5, 1, 12, 0, tzinfo=TZ) + timedelta(minutes=start_offset)
    return SacctRow(
        job_id=job_id,
        user=user,
        submit_time=submit,
        start_time=start,
        state="COMPLETED",
        partition=partition,
        nodes=nodes,
        alloc_gres=alloc_gres,
    )


def test_filter_rows_excludes_steps_and_filters_users_partitions():
    rows = [
        make_row("123", 0, 10, user="alice", partition="gpu-a", nodes=1, alloc_gres="gpu:1"),
        make_row("123.batch", 0, 10, user="alice", partition="gpu-a", nodes=1, alloc_gres="gpu:1"),
        make_row("456", 0, 5, user="bob", partition="gpu-b", nodes=1, alloc_gres="gpu:1"),
        make_row("789", 0, 15, user="carol", partition="cpu", nodes=1, alloc_gres=None),
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


def test_determine_job_type_infers_expected_categories():
    cpu_row = make_row("1", 0, 5, alloc_gres=None)
    assert determine_job_type(cpu_row) == "cpu-only"

    single_gpu_row = make_row("2", 0, 5, alloc_gres="gpu:1")
    assert determine_job_type(single_gpu_row) == "1-gpu"

    multi_gpu_row = make_row("3", 0, 5, alloc_gres="gpu:tesla:4", nodes=1)
    assert determine_job_type(multi_gpu_row) == "single-node"

    multi_node_row = make_row("4", 0, 5, alloc_gres="gpu:4", nodes=2)
    assert determine_job_type(multi_node_row) == "multi-node"


def test_filter_rows_supports_job_type_filter():
    rows = [
        make_row("cpu", 0, 5, alloc_gres=None),
        make_row("one-gpu", 0, 5, alloc_gres="gpu:1"),
        make_row("multi-gpu", 0, 5, alloc_gres="gpu:4", nodes=1),
        make_row("multi-node", 0, 5, alloc_gres="gpu:1", nodes=3),
    ]

    filtered = filter_rows(rows, job_type="1-gpu")
    assert [record.job_id for record in filtered] == ["one-gpu"]
    assert filtered[0].job_type == "1-gpu"
