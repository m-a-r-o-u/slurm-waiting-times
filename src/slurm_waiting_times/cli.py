from __future__ import annotations

import argparse
import logging
import shlex
import sys
from datetime import datetime, timedelta
from statistics import mean
from typing import Sequence

from .histogram import create_histogram
from .output import build_prefix, histogram_path, results_csv_path, write_results_csv
from .processing import filter_rows
from .sacct import SacctError, build_sacct_command, parse_sacct_output, run_sacct
from .time_utils import (
    ensure_timezone,
    format_timedelta_hms,
    parse_cli_datetime_window,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_WINDOW_DAYS = 14


class CliError(RuntimeError):
    """Raised when CLI validation fails."""


def _split_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _has_wildcard(patterns: Sequence[str]) -> bool:
    return any(any(ch in pattern for ch in "*?[") for pattern in patterns)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and visualise Slurm job waiting times.",
    )
    parser.add_argument("--start", help="Start datetime for the sacct query.")
    parser.add_argument("--end", help="End datetime for the sacct query.")
    parser.add_argument("--user", help="Comma-separated list of users to include.")
    parser.add_argument(
        "--partition",
        help="Comma-separated list of partitions to include. Wildcards are supported.",
    )
    parser.add_argument(
        "--include-steps",
        action="store_true",
        help="Include job steps such as .batch and .extern entries.",
    )
    parser.add_argument("--tz", help="IANA timezone to interpret timestamps.")
    parser.add_argument("--dry-run", action="store_true", help="Print the sacct command and exit.")
    parser.add_argument(
        "--bins",
        type=int,
        help="Explicit number of histogram bins to use.",
    )
    parser.add_argument(
        "--bin-seconds",
        action="store_true",
        help="Use seconds instead of minutes on the histogram X-axis.",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=float,
        help="Discard jobs whose waiting time exceeds this number of hours.",
    )

    return parser.parse_args(argv)


def _validate_bins(bins: int | None) -> int | None:
    if bins is None:
        return None
    if bins <= 0:
        raise CliError("--bins must be a positive integer")
    return bins


def _validate_max_wait(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        raise CliError("--max-wait-hours must be greater than zero")
    return value


def _prepare_filters(user_arg: str | None, partition_arg: str | None) -> tuple[list[str] | None, list[str] | None]:
    users = _split_arg(user_arg)
    partitions = _split_arg(partition_arg)
    return users, partitions


def _format_datetime_for_token(value: datetime) -> str:
    if value.second or value.microsecond:
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if value.hour == 0 and value.minute == 0:
        return value.strftime("%Y-%m-%d")
    return value.strftime("%Y-%m-%dT%H:%M")


def _args_tokens(
    *,
    start_supplied: bool,
    start_value: datetime,
    end_value: datetime,
    users: Sequence[str] | None,
    partitions: Sequence[str] | None,
    include_steps: bool,
    tz: str | None,
    bins: int | None,
    bin_seconds: bool,
    max_wait_hours: float | None,
) -> list[str]:
    tokens: list[str] = []
    if start_supplied:
        tokens.append(f"start={_format_datetime_for_token(start_value)}")
    tokens.append(f"end={_format_datetime_for_token(end_value)}")
    if users:
        tokens.append(f"user={','.join(users)}")
    else:
        tokens.append("user=all")
    if partitions:
        tokens.append(f"partition={','.join(partitions)}")
    if include_steps:
        tokens.append("steps")
    # Timezone is intentionally omitted from the filename prefix for clarity.
    if bins is not None:
        tokens.append(f"bins={bins}")
    if bin_seconds:
        tokens.append("seconds")
    if max_wait_hours is not None:
        tokens.append(f"maxwait={max_wait_hours:g}")
    return tokens


def _title(
    *,
    start: datetime,
    end: datetime,
    users: Sequence[str] | None,
    partitions: Sequence[str] | None,
    include_steps: bool,
) -> str:
    user_summary = ",".join(users) if users else "all users"
    partition_summary = ",".join(partitions) if partitions else "all partitions"
    steps_summary = "steps included" if include_steps else None
    details = [user_summary, partition_summary]
    if steps_summary:
        details.append(steps_summary)
    return (
        "Waiting times "
        f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} "
        f"({'; '.join(details)})"
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_arguments(argv)
        bins = _validate_bins(args.bins)
        max_wait = _validate_max_wait(args.max_wait_hours)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        tzinfo = ensure_timezone(args.tz)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    now = datetime.now(tzinfo)
    default_end = now
    default_start = now - timedelta(days=DEFAULT_WINDOW_DAYS)

    try:
        start_dt, end_dt = parse_cli_datetime_window(
            args.start,
            args.end,
            default_start,
            default_end,
            tzinfo,
        )
    except ValueError as exc:
        print(f"Error parsing date: {exc}", file=sys.stderr)
        return 2

    if start_dt > end_dt:
        print("Error: --start must be before --end", file=sys.stderr)
        return 2

    users, partitions = _prepare_filters(args.user, args.partition)

    command_users = users if users and not _has_wildcard(users) else None
    command_partitions = partitions if partitions and not _has_wildcard(partitions) else None

    command = build_sacct_command(
        start_dt,
        end_dt,
        users=command_users,
        partitions=command_partitions,
        include_steps=args.include_steps,
    )

    if args.dry_run:
        print(shlex.join(command))
        return 0

    try:
        sacct_output = run_sacct(command)
    except SacctError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    rows = parse_sacct_output(sacct_output, timezone=args.tz)
    records = filter_rows(
        rows,
        include_steps=args.include_steps,
        user_filters=users,
        partition_filters=partitions,
        max_wait_hours=max_wait,
    )

    if not records:
        print("No jobs found in the specified window.", file=sys.stderr)
        return 1

    mean_wait_seconds = mean(record.wait_seconds for record in records)
    summary = (
        f"Jobs: {len(records)} | Window: {start_dt.isoformat()} -> {end_dt.isoformat()} "
        f"| Mean wait: {format_timedelta_hms(mean_wait_seconds)}"
    )
    print(summary)

    tokens = _args_tokens(
        start_supplied=args.start is not None,
        start_value=start_dt,
        end_value=end_dt,
        users=users,
        partitions=partitions,
        include_steps=args.include_steps,
        tz=args.tz,
        bins=bins,
        bin_seconds=args.bin_seconds,
        max_wait_hours=max_wait,
    )
    prefix = build_prefix(now, tokens)

    csv_path = results_csv_path(prefix)
    write_results_csv(csv_path, records)

    fig = create_histogram(
        records,
        use_seconds=args.bin_seconds,
        bins=bins,
        title=_title(
            start=start_dt,
            end=end_dt,
            users=users,
            partitions=partitions,
            include_steps=args.include_steps,
        ),
    )
    fig.savefig(histogram_path(prefix))
    fig.clf()
    try:
        from matplotlib import pyplot as plt
    except Exception:  # pragma: no cover - optional dependency may be missing
        pass
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
