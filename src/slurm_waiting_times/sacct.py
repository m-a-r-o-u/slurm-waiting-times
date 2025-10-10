from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from typing import Iterable, List, Sequence

from .models import SacctRow
from .time_utils import ensure_timezone, parse_datetime

LOGGER = logging.getLogger(__name__)


SACCT_FORMAT = "JobID,User,Submit,Start,State,Partition,NNodes,AllocTRES"
INVALID_START_VALUES = {"unknown", "none", "", "n/a", "invalid"}
EMPTY_FIELD_VALUES = {"", "none", "n/a", "unknown", "(null)"}


class SacctError(RuntimeError):
    """Raised when sacct execution fails."""


def build_sacct_command(
    start: datetime,
    end: datetime,
    *,
    users: Sequence[str] | None = None,
    partitions: Sequence[str] | None = None,
    include_steps: bool = False,
) -> List[str]:
    command = [
        "sacct",
        "--parsable2",
        "--noheader",
        f"--format={SACCT_FORMAT}",
        "-S",
        start.strftime("%Y-%m-%dT%H:%M:%S"),
        "-E",
        end.strftime("%Y-%m-%dT%H:%M:%S"),
    ]

    if users:
        command.extend(["--user", ",".join(users)])
    else:
        command.append("-a")

    if not include_steps:
        command.append("-X")

    if partitions:
        command.extend(["--partition", ",".join(partitions)])

    return command


def run_sacct(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SacctError("sacct command not found") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise SacctError(
            f"sacct returned non-zero exit code {exc.returncode}: {exc.stderr.strip()}"
        ) from exc

    return result.stdout


def parse_sacct_output(
    output: str,
    *,
    timezone: str | None = None,
) -> List[SacctRow]:
    tzinfo = ensure_timezone(timezone)
    rows: List[SacctRow] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("|")
        if len(parts) != 8:
            LOGGER.warning("Skipping malformed sacct row: %s", raw_line)
            continue

        (
            job_id,
            user,
            submit,
            start,
            state,
            partition,
            raw_nodes,
            alloc_tres,
        ) = parts
        if start.strip().lower() in INVALID_START_VALUES:
            LOGGER.debug("Dropping job %s due to invalid start value '%s'", job_id, start)
            continue

        try:
            submit_dt = parse_datetime(submit, tzinfo)
            start_dt = parse_datetime(start, tzinfo)
        except ValueError as exc:
            LOGGER.warning("Skipping job %s because of timestamp error: %s", job_id, exc)
            continue

        nodes = None
        raw_nodes_stripped = raw_nodes.strip()
        if raw_nodes_stripped and raw_nodes_stripped.lower() not in EMPTY_FIELD_VALUES:
            try:
                nodes = int(raw_nodes_stripped)
            except ValueError:
                LOGGER.debug("Unable to parse node count '%s' for job %s", raw_nodes, job_id)

        alloc_tres_value = alloc_tres.strip() or None
        if alloc_tres_value and alloc_tres_value.lower() in EMPTY_FIELD_VALUES:
            alloc_tres_value = None

        rows.append(
            SacctRow(
                job_id=job_id,
                user=user,
                submit_time=submit_dt,
                start_time=start_dt,
                state=state,
                partition=partition,
                nodes=nodes,
                alloc_tres=alloc_tres_value,
            )
        )

    return rows
