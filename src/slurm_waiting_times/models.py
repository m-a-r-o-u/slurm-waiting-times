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
    nodes: int | None
    alloc_tres: str | None
    elapsed_seconds: float | None

    @property
    def alloc_gres(self) -> str | None:  # pragma: no cover - compatibility shim
        """Backward compatible alias for the removed AllocGRES field."""

        return self.alloc_tres

    @alloc_gres.setter  # pragma: no cover - compatibility shim
    def alloc_gres(self, value: str | None) -> None:
        self.alloc_tres = value


@dataclass(slots=True)
class JobRecord(SacctRow):
    """A sacct row augmented with waiting-time metadata."""

    wait_seconds: float
    job_type: str | None = None
