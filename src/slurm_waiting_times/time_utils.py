from __future__ import annotations

import math
import calendar
import re
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


_MONTH_ONLY_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$")


def _month_bounds(year: int, month: int, tzinfo: ZoneInfo) -> tuple[datetime, datetime]:
    """Return the first and last instants of ``year``-``month`` in ``tzinfo``."""

    first = datetime(year, month, 1, tzinfo=tzinfo)
    last_day = calendar.monthrange(year, month)[1]
    last = datetime(year, month, last_day, 23, 59, 59, tzinfo=tzinfo)
    return first, last


def _parse_month(value: str, tzinfo: ZoneInfo) -> tuple[datetime, datetime]:
    match = _MONTH_ONLY_PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"Unrecognised month format: '{value}'")
    year = int(match.group("year"))
    month = int(match.group("month"))
    if not 1 <= month <= 12:
        raise ValueError(f"Unrecognised month format: '{value}'")
    return _month_bounds(year, month, tzinfo)


def parse_cli_datetime(value: str | None, default: datetime, tzinfo: ZoneInfo) -> datetime:
    """Parse a CLI datetime argument, falling back to ``default``."""

    if value is None:
        return default.astimezone(tzinfo)

    return parse_datetime(value, tzinfo)


def parse_cli_datetime_window(
    start_value: str | None,
    end_value: str | None,
    default_start: datetime,
    default_end: datetime,
    tzinfo: ZoneInfo,
) -> tuple[datetime, datetime]:
    """Parse CLI datetime arguments with optional month-based shortcuts."""

    default_start = default_start.astimezone(tzinfo)
    default_end = default_end.astimezone(tzinfo)

    inferred_end: datetime | None = None

    if start_value is None:
        start_dt = default_start
    else:
        stripped = start_value.strip()
        if _MONTH_ONLY_PATTERN.match(stripped):
            start_dt, inferred_end = _parse_month(stripped, tzinfo)
        else:
            start_dt = parse_datetime(stripped, tzinfo)

    if end_value is None:
        end_dt = inferred_end if inferred_end is not None else default_end
    else:
        stripped = end_value.strip()
        if _MONTH_ONLY_PATTERN.match(stripped):
            _, end_dt = _parse_month(stripped, tzinfo)
        else:
            end_dt = parse_datetime(stripped, tzinfo)

    return start_dt, end_dt


def format_timedelta_hms(seconds: float) -> str:
    """Format ``seconds`` as a human readable duration without seconds."""

    total_minutes = int(math.floor(seconds / 60.0 + 0.5))
    if total_minutes < 0:
        total_minutes = 0

    days, remainder_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(remainder_minutes, 60)

    return f"{days}-{hours:02d}:{minutes:02d}"


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
