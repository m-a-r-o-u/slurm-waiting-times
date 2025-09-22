from datetime import datetime

from slurm_waiting_times.output import build_prefix, compact_args


def test_compact_args_sanitises_and_truncates():
    tokens = ["user=alice bob", "partition=gpu*a"]
    compact = compact_args(tokens)
    assert compact.startswith("user=alice_bob")
    assert "*" not in compact


def test_build_prefix_includes_timestamp():
    now = datetime(2024, 5, 1, 12, 30)
    prefix = build_prefix(now, ["start=20240501"])
    assert prefix.startswith("2024-05-01_12:30-start=20240501")
