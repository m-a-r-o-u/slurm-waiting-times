from datetime import datetime

from slurm_waiting_times.sacct import build_sacct_command, parse_sacct_output


def test_parse_sacct_output_skips_invalid_rows():
    output = """
123|alice|2024-05-01T10:00:00|2024-05-01T10:05:00|COMPLETED|debug
456|bob|2024-05-02T11:00:00|Unknown|PENDING|gpu
789|carol|2024-05-03T12:00:00|2024-05-03T12:20:00|FAILED|gpu
    """.strip()

    rows = parse_sacct_output(output, timezone="UTC")
    assert len(rows) == 2
    assert rows[0].job_id == "123"
    assert rows[0].user == "alice"
    assert rows[0].submit_time.isoformat() == "2024-05-01T10:00:00+00:00"
    assert rows[1].job_id == "789"


def test_parse_sacct_output_warns_on_bad_timestamp(caplog):
    bad_output = "123|alice|not-a-time|2024-05-01T10:05:00|COMPLETED|debug"
    with caplog.at_level("WARNING"):
        rows = parse_sacct_output(bad_output, timezone="UTC")
    assert rows == []
    assert "timestamp error" in caplog.text


def test_build_sacct_command_includes_all_users_flag_by_default():
    start = datetime(2025, 9, 15)
    end = datetime(2025, 9, 22)

    command = build_sacct_command(start, end)

    assert "-a" in command
    assert "--user" not in command


def test_build_sacct_command_uses_specific_users_when_requested():
    start = datetime(2025, 9, 15)
    end = datetime(2025, 9, 22)

    command = build_sacct_command(start, end, users=["alice", "bob"])

    assert "-a" not in command
    assert command[command.index("--user") + 1] == "alice,bob"
