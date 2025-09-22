from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo


def ensure_timezone(tz_name: str | None) -> ZoneInfo:
    """Return a :class:`~zoneinfo.ZoneInfo` instance for ``tz_name``.

    If ``tz_name`` is ``None`` the system local timezone is used.  When the
    local timezone cannot be determined the function falls back to UTC.
    """

    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except Exception as exc:  # pragma: no cover - only raised on invalid tz names
            raise ValueError(f"Unknown timezone '{tz_name}'") from exc

    local = datetime.now().astimezone().tzinfo
    if local is None:
        return ZoneInfo("UTC")
    if isinstance(local, ZoneInfo):
        return local
    # ``astimezone`` may return a different tzinfo implementation.  ZoneInfo can
    # build an equivalent representation via the key.
    try:
        return ZoneInfo(str(local.key))  # type: ignore[attr-defined]
    except Exception:
        return ZoneInfo("UTC")


def parse_datetime(value: str, tzinfo: ZoneInfo) -> datetime:
    """Parse a Slurm timestamp and normalise it to ``tzinfo``.

    The parser accepts standard ISO-8601 timestamps and the common Slurm
    variants that omit seconds.  When the parsed value lacks timezone
    information the supplied ``tzinfo`` is attached.
    """

    value = value.strip()
    if not value:
        raise ValueError("empty datetime value")

    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        for pattern in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
        ):
            try:
                dt = datetime.strptime(value, pattern)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognised datetime format: '{value}'")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzinfo)
    else:
        dt = dt.astimezone(tzinfo)

    return dt


def parse_cli_datetime(value: str | None, default: datetime, tzinfo: ZoneInfo) -> datetime:
    """Parse a CLI datetime argument, falling back to ``default``."""

    if value is None:
        return default.astimezone(tzinfo)

    return parse_datetime(value, tzinfo)


def format_timedelta_hms(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def freedman_diaconis_bins(values: Iterable[float]) -> int:
    data = [float(v) for v in values]
    n = len(data)
    if n == 0:
        raise ValueError("freedman_diaconis_bins() requires at least one data point")

    if n == 1:
        return 1

    data.sort()

    def percentile(p: float) -> float:
        if not 0 <= p <= 1:
            raise ValueError("percentile must be between 0 and 1")
        idx = (n - 1) * p
        lower = math.floor(idx)
        upper = math.ceil(idx)
        if lower == upper:
            return data[int(idx)]
        lower_value = data[lower]
        upper_value = data[upper]
        return lower_value + (upper_value - lower_value) * (idx - lower)

    q1 = percentile(0.25)
    q3 = percentile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return max(1, int(round(n ** 0.5)))

    width = 2 * iqr / (n ** (1 / 3))
    data_range = data[-1] - data[0]
    if data_range == 0:
        return 1

    return max(1, math.ceil(data_range / width))
