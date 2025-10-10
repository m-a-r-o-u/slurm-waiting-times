from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .models import JobRecord

_OUTPUT_DIR = Path("output")
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9,._=-]")


def ensure_output_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


def compact_args(tokens: Sequence[str]) -> str | None:
    if not tokens:
        return None

    joined = "_".join(token.replace(" ", "_") for token in tokens)
    sanitised = _SANITIZE_RE.sub("_", joined)
    if len(sanitised) > 80:
        sanitised = sanitised[:80]
    return sanitised


def build_prefix(now: datetime, tokens: Sequence[str]) -> str:
    args_part = compact_args(tokens)
    if args_part:
        return args_part
    return now.strftime("%Y-%m-%d")


def results_csv_path(prefix: str) -> Path:
    return ensure_output_dir() / f"{prefix}-waiting-times.csv"


def histogram_path(prefix: str) -> Path:
    return ensure_output_dir() / f"{prefix}-waiting-times.png"


def write_results_csv(path: Path, records: Iterable[JobRecord]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "JobID",
                "User",
                "Submit",
                "Start",
                "State",
                "Partition",
                "Nodes",
                "AllocGRES",
                "JobType",
                "WaitSeconds",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.job_id,
                    record.user,
                    record.submit_time.isoformat(),
                    record.start_time.isoformat(),
                    record.state,
                    record.partition,
                    "" if record.nodes is None else record.nodes,
                    record.alloc_gres or "",
                    record.job_type or "",
                    f"{record.wait_seconds:.2f}",
                ]
            )
