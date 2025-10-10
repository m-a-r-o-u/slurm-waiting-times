from datetime import datetime

from slurm_waiting_times.cli import _args_tokens, _title


def test_args_tokens_include_end_token_when_not_explicitly_supplied():
    tokens = _args_tokens(
        start_supplied=True,
        start_value=datetime(2025, 3, 1),
        end_value=datetime(2025, 3, 31),
        users=None,
        partitions=None,
        include_steps=False,
        tz=None,
        bins=None,
        bin_seconds=False,
        max_wait_hours=None,
        job_type=None,
    )

    assert tokens[0] == "start=2025-03-01"
    assert tokens[1] == "end=2025-03-31"


def test_args_tokens_include_job_type_when_requested():
    tokens = _args_tokens(
        start_supplied=False,
        start_value=datetime(2025, 3, 1),
        end_value=datetime(2025, 3, 31),
        users=None,
        partitions=None,
        include_steps=False,
        tz=None,
        bins=None,
        bin_seconds=False,
        max_wait_hours=None,
        job_type="multi-node",
    )

    assert "jobtype=multi-node" in tokens


def test_title_includes_job_type_when_requested():
    title = _title(
        start=datetime(2025, 3, 1),
        end=datetime(2025, 3, 31),
        users=None,
        partitions=None,
        include_steps=False,
        job_type="1-gpu",
    )

    assert "(all users; all partitions; 1-gpu)" in title
