from datetime import datetime

import pytest

from slurm_waiting_times.cli import CliError, _args_tokens, _parse_runtime_filters, _title


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
        runtime_filters=None,
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
        runtime_filters=None,
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


def test_args_tokens_include_runtime_filters():
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
        job_type=None,
        runtime_filters=[">01:00:00", "01:00:00-02:00:00"],
    )

    assert "runtime=>01:00:00" in tokens
    assert "runtime=01:00:00-02:00:00" in tokens


def test_parse_runtime_filters_supports_comparisons_and_ranges():
    filters = _parse_runtime_filters([">=00:30:00", "<02:00:00"])

    assert len(filters) == 2
    assert all(f.matches(3600) for f in filters)
    assert not all(f.matches(1200) for f in filters)
    assert not all(f.matches(7200) for f in filters)


def test_parse_runtime_filters_supports_words():
    filters = _parse_runtime_filters(["shorter:01:00:00", "longer:00:10:00"])

    assert len(filters) == 2
    assert all(f.matches(1800) for f in filters)
    assert not all(f.matches(30) for f in filters)


def test_parse_runtime_filters_rejects_invalid_values():
    with pytest.raises(CliError):
        _parse_runtime_filters(["invalid"])
