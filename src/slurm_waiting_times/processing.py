from __future__ import annotations

import fnmatch
from typing import Iterable, List, Sequence

from .models import JobRecord, SacctRow


def _matches(value: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def filter_rows(
    rows: Iterable[SacctRow],
    *,
    include_steps: bool = False,
    user_filters: Sequence[str] | None = None,
    partition_filters: Sequence[str] | None = None,
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
                wait_seconds=wait_seconds,
            )
        )

    return filtered
