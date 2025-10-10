from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .models import JobRecord, SacctRow


def _matches(value: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


_GPU_PATTERN = re.compile(r"gpu(?::[^,()]+)*:(\d+)", flags=re.IGNORECASE)


def _count_gpus(alloc_tres: str | None) -> int:
    if not alloc_tres:
        return 0

    cleaned = re.sub(r"\([^)]*\)", "", alloc_tres)

    gpu_total = 0
    has_tres_format = False
    for chunk in cleaned.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.lower().startswith("gres/gpu"):
            continue

        has_tres_format = True
        if "=" not in chunk:
            continue

        _, value = chunk.split("=", 1)
        value = value.strip()
        if not value:
            continue

        try:
            gpu_total += int(value)
            continue
        except ValueError:
            pass

        match = re.search(r"(\d+)", value)
        if match:
            gpu_total += int(match.group(1))

    if has_tres_format:
        return gpu_total

    matches = _GPU_PATTERN.findall(cleaned)
    return sum(int(match) for match in matches)


@dataclass(frozen=True)
class RuntimeConstraint:
    min_seconds: float | None = None
    max_seconds: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True

    def matches(self, value: float) -> bool:
        if self.min_seconds is not None:
            if value < self.min_seconds:
                return False
            if not self.min_inclusive and value == self.min_seconds:
                return False
        if self.max_seconds is not None:
            if value > self.max_seconds:
                return False
            if not self.max_inclusive and value == self.max_seconds:
                return False
        return True


def determine_job_type(row: SacctRow) -> str | None:
    """Infer the job type from sacct metadata."""

    nodes = row.nodes
    gpu_count = _count_gpus(row.alloc_tres)

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
    runtime_filters: Sequence[RuntimeConstraint] | None = None,
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

        if runtime_filters:
            if row.elapsed_seconds is None:
                continue
            if not all(constraint.matches(row.elapsed_seconds) for constraint in runtime_filters):
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
                alloc_tres=row.alloc_tres,
                elapsed_seconds=row.elapsed_seconds,
                wait_seconds=wait_seconds,
                job_type=row_job_type,
            )
        )

    return filtered
