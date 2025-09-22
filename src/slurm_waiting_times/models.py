from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class SacctRow:
    """Representation of a row emitted by ``sacct``."""

    job_id: str
    user: str
    submit_time: datetime
    start_time: datetime
    state: str
    partition: str


@dataclass(slots=True)
class JobRecord(SacctRow):
    """A sacct row augmented with waiting-time metadata."""

    wait_seconds: float
