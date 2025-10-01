from datetime import datetime

from slurm_waiting_times.output import build_prefix, compact_args


def test_compact_args_sanitises_and_truncates():
    tokens = ["user=alice bob", "partition=gpu*a"]
    compact = compact_args(tokens)
    assert compact.startswith("user=alice_bob")
    assert "*" not in compact


def test_build_prefix_uses_tokens_without_timestamp():
    now = datetime(2024, 5, 1, 12, 30)
    prefix = build_prefix(
        now,
        [
            "start=2024-05-01",
            "end=2024-05-31",
            "user=all",
        ],
    )
    assert prefix == "start=2024-05-01_end=2024-05-31_user=all"


def test_build_prefix_falls_back_to_date():
    now = datetime(2024, 5, 1, 12, 30)
    prefix = build_prefix(now, [])
    assert prefix == "2024-05-01"
