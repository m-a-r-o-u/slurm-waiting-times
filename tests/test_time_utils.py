from datetime import datetime

import pytest
from zoneinfo import ZoneInfo

from slurm_waiting_times.time_utils import (
    format_timedelta_hms,
    freedman_diaconis_bins,
    parse_datetime,
    parse_cli_datetime_window,
    parse_duration_to_seconds,
)


def test_freedman_diaconis_bins_varied_data():
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    bins = freedman_diaconis_bins(values)
    assert bins >= 1


def test_freedman_diaconis_bins_single_value():
    assert freedman_diaconis_bins([5]) == 1


@pytest.mark.parametrize(
    "value",
    [
        "2024-05-01T12:34:56",
        "2024-05-01 12:34:56",
        "2024/05/01 12:34:56",
    ],
)
def test_parse_datetime_accepts_multiple_formats(value):
    tz = ZoneInfo("UTC")
    dt = parse_datetime(value, tz)
    assert dt.year == 2024
    assert dt.tzinfo == tz
@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00"),
        (29, "00:00"),
        (30, "00:01"),
        (89, "00:01"),
        (90, "00:02"),
        (3661, "01:01"),
        ((24 * 60 * 60) + (2 * 60 * 60) + (30 * 60), "1-02:30"),
    ],
)
def test_format_timedelta_hms(seconds, expected):
    assert format_timedelta_hms(seconds) == expected


def test_parse_cli_datetime_window_month_only_start_and_default_end():
    tz = ZoneInfo("UTC")
    default_start = datetime(2024, 1, 15, tzinfo=tz)
    default_end = datetime(2024, 1, 31, tzinfo=tz)

    start, end = parse_cli_datetime_window("2025-09", None, default_start, default_end, tz)

    assert start == datetime(2025, 9, 1, tzinfo=tz)
    assert end == datetime(2025, 9, 30, 23, 59, 59, tzinfo=tz)


def test_parse_cli_datetime_window_month_only_start_and_end():
    tz = ZoneInfo("UTC")
    default_start = datetime(2024, 1, 15, tzinfo=tz)
    default_end = datetime(2024, 1, 31, tzinfo=tz)

    start, end = parse_cli_datetime_window("2025-09", "2025-12", default_start, default_end, tz)

    assert start == datetime(2025, 9, 1, tzinfo=tz)
    assert end == datetime(2025, 12, 31, 23, 59, 59, tzinfo=tz)


def test_parse_cli_datetime_window_default_start_and_month_only_end():
    tz = ZoneInfo("UTC")
    default_start = datetime(2024, 1, 15, tzinfo=tz)
    default_end = datetime(2024, 1, 31, tzinfo=tz)

    start, end = parse_cli_datetime_window(None, "2025-12", default_start, default_end, tz)

    assert start == default_start
    assert end == datetime(2025, 12, 31, 23, 59, 59, tzinfo=tz)


def test_parse_cli_datetime_window_invalid_month():
    tz = ZoneInfo("UTC")
    default_start = datetime(2024, 1, 15, tzinfo=tz)
    default_end = datetime(2024, 1, 31, tzinfo=tz)

    with pytest.raises(ValueError):
        parse_cli_datetime_window("2025-13", None, default_start, default_end, tz)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("00:00:30", 30),
        ("01:02:03", (1 * 3600) + (2 * 60) + 3),
        ("2-03:00:00", (2 * 24 + 3) * 3600),
    ],
)
def test_parse_duration_to_seconds_valid(value, expected):
    assert parse_duration_to_seconds(value) == expected


@pytest.mark.parametrize("value", ["", "1:60:00", "not-a-duration"])
def test_parse_duration_to_seconds_invalid(value):
    with pytest.raises(ValueError):
        parse_duration_to_seconds(value)
