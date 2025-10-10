from datetime import datetime

from slurm_waiting_times.sacct import build_sacct_command, parse_sacct_output


def test_parse_sacct_output_skips_invalid_rows():
    output = "\n".join(
        [
            "123|123|job-a|sbatch job-a.sh|alice|2024-05-01T10:00:00|2024-05-01T10:05:00|COMPLETED|debug|1|cpu=4,mem=8G,node=1,gres/gpu=1|00:30:00",
            "456|456|job-b|srun --pty bash|bob|2024-05-02T11:00:00|Unknown|PENDING|gpu|2|cpu=8,mem=16G,node=1,gres/gpu=4|00:45:00",
            "789|789|job-c||carol|2024-05-03T12:00:00|2024-05-03T12:20:00|FAILED|gpu|4|cpu=32,mem=64G,node=2,gres/gpu=8|1-01:00:00",
        ]
    )

    rows = parse_sacct_output(output, timezone="UTC")
    assert len(rows) == 2
    assert rows[0].job_id == "123"
    assert rows[0].job_id_raw == "123"
    assert rows[0].job_name == "job-a"
    assert rows[0].submit_line == "sbatch job-a.sh"
    assert rows[0].user == "alice"
    assert rows[0].submit_time.isoformat() == "2024-05-01T10:00:00+00:00"
    assert rows[0].nodes == 1
    assert rows[0].alloc_tres == "cpu=4,mem=8G,node=1,gres/gpu=1"
    assert rows[0].elapsed_seconds == 1800
    assert rows[1].job_id == "789"
    assert rows[1].nodes == 4
    assert rows[1].alloc_tres == "cpu=32,mem=64G,node=2,gres/gpu=8"
    assert rows[1].elapsed_seconds == ((1 * 24 + 1) * 3600)


def test_parse_sacct_output_warns_on_bad_timestamp(caplog):
    bad_output = (
        "123|123|job|sbatch script.sh|alice|not-a-time|2024-05-01T10:05:00|COMPLETED|debug|1|cpu=1,gres/gpu=1|00:10:00"
    )
    with caplog.at_level("WARNING"):
        rows = parse_sacct_output(bad_output, timezone="UTC")
    assert rows == []
    assert "timestamp error" in caplog.text


def test_parse_sacct_output_handles_multiline_submit_line():
    output = "\n".join(
        [
            "5300915|5300915|interactive|salloc -p debug --time=01:00:00 --pty bash -i -c",
            "      echo 'line one'",
            "      echo 'line two'",
            "    |carol|2025-08-30T17:40:10|2025-08-30T17:45:10|COMPLETED|gpu|1|cpu=4,mem=16G|00:05:00",
        ]
    )

    rows = parse_sacct_output(output, timezone="UTC")

    assert len(rows) == 1
    assert rows[0].job_id == "5300915"
    assert rows[0].user == "carol"
    assert rows[0].submit_line == (
        "salloc -p debug --time=01:00:00 --pty bash -i -c\n"
        "      echo 'line one'\n"
        "      echo 'line two'"
    )


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
