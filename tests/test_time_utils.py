import pytest
import pytest
from zoneinfo import ZoneInfo

from slurm_waiting_times.time_utils import (
    format_timedelta_hms,
    freedman_diaconis_bins,
    parse_datetime,
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


def test_format_timedelta_hms():
    assert format_timedelta_hms(3661) == "01:01:01"
