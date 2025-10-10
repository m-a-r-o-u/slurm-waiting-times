from __future__ import annotations

import fnmatch
import re
from typing import Iterable, List, Sequence

from .models import JobRecord, SacctRow


def _matches(value: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


_GPU_PATTERN = re.compile(r"gpu(?::[^,()]+)*:(\d+)", flags=re.IGNORECASE)


def _count_gpus(alloc_gres: str | None) -> int:
    if not alloc_gres:
        return 0

    cleaned = re.sub(r"\([^)]*\)", "", alloc_gres)
    matches = _GPU_PATTERN.findall(cleaned)
    return sum(int(match) for match in matches)


def determine_job_type(row: SacctRow) -> str | None:
    """Infer the job type from sacct metadata."""

    nodes = row.nodes
    gpu_count = _count_gpus(row.alloc_gres)

    if nodes is not None and nodes > 1:
        return "multi-node"

    if gpu_count == 0:
        return "cpu-only"

    if gpu_count == 1:
        return "1-gpu"

    if nodes == 1 or nodes is None:
        return "single-node"

    return None


def filter_rows(
    rows: Iterable[SacctRow],
    *,
    include_steps: bool = False,
    user_filters: Sequence[str] | None = None,
    partition_filters: Sequence[str] | None = None,
    job_type: str | None = None,
    max_wait_hours: float | None = None,
) -> List[JobRecord]:
    filtered: List[JobRecord] = []
    wait_cap = None if max_wait_hours is None else max_wait_hours * 3600

    for row in rows:
        if not include_steps and "." in row.job_id:
            continue

        if user_filters and not _matches(row.user, user_filters):
            continue

        if partition_filters and not _matches(row.partition, partition_filters):
            continue

        wait_seconds = (row.start_time - row.submit_time).total_seconds()

        row_job_type = determine_job_type(row)

        if job_type and row_job_type != job_type:
            continue

        if wait_cap is not None and wait_seconds > wait_cap:
            continue

        filtered.append(
            JobRecord(
                job_id=row.job_id,
                user=row.user,
                submit_time=row.submit_time,
                start_time=row.start_time,
                state=row.state,
                partition=row.partition,
                nodes=row.nodes,
                alloc_gres=row.alloc_gres,
                wait_seconds=wait_seconds,
                job_type=row_job_type,
            )
        )

    return filtered
