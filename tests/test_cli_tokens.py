from datetime import datetime

from slurm_waiting_times.cli import _args_tokens


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
    )

    assert tokens[0] == "start=2025-03-01"
    assert tokens[1] == "end=2025-03-31"
