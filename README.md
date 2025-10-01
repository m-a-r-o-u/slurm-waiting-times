# slurm-waiting-times

`slurm-waiting-times` queries `sacct`, filters the returned jobs, and visualises queueing delays so that operators can reason about cluster health. Waiting time is defined as `Start - Submit`, array tasks are treated as independent jobs, and step records such as `.batch`/`.extern` are hidden by default.

## Requirements

* Python 3.10+
* Access to a Slurm installation with `sacct`
* [uv](https://github.com/astral-sh/uv) for managing the virtual environment

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```text
slurm-waiting-times [--start <time>] [--end <time>] [--user <list>] [--partition <list>] \
                    [--include-steps] [--tz <zone>] [--bins <n>] [--bin-seconds] \
                    [--max-wait-hours <hours>] [--dry-run]
```

* `--start` / `--end`: ISO or Slurm-style datetimes. Defaults to the last 14 days ending “now”.
  Supplying a year and month (e.g. `2025-09`) expands to the first or last day of
  that month as appropriate, allowing quick whole-month reports.
* `--user`: comma-separated users to include. Array-task IDs (e.g. `12345_7`) are kept, job steps (`12345.batch`) are dropped unless `--include-steps` is present.
* `--partition`: comma-separated list of partitions; shell-style wildcards are accepted.
* `--tz`: interpret timestamps in the supplied IANA timezone (defaults to the local system zone).
* `--bins`: override the Freedman–Diaconis bin selection.
* `--bin-seconds`: express waiting times in seconds rather than minutes on the histogram X-axis.
* `--max-wait-hours`: discard outliers above the supplied waiting time.
* `--dry-run`: print the `sacct` command instead of executing it.

When the query returns jobs, the CLI prints a summary line containing the job count, effective window, and mean waiting time (HH:MM:SS). Detailed results and the histogram are written to `output/` as:

* `YYYY-MM-DD_HH:MM-<args>-waiting-times.csv`
* `YYYY-MM-DD_HH:MM-<args>-waiting-times.png`

The file prefix contains only non-default arguments (spaces are replaced with underscores and the string is truncated to 40 characters).

## Examples

```bash
slurm-waiting-times --start 2025-09-15 --end 2025-09-22
slurm-waiting-times --user alice --partition lrz*
slurm-waiting-times --user alice,bob --partition mcml-a100,mcml-h100
```

## Output interpretation

The CSV file contains job metadata plus a `WaitSeconds` column. The histogram uses minutes by default, adds a dashed red line at the mean waiting time, and includes a legend annotation. All timestamps are normalised to the selected timezone.

To inspect a histogram, run the CLI with your desired arguments and open the generated PNG in the `output/` directory.

## Testing

```bash
uv pip install -e .[test]
pytest
```

## Notes

* Jobs without a valid `Start` timestamp (e.g. `Unknown`, `None`) are excluded.
* Array tasks are considered first-class jobs.
* Steps can be included with `--include-steps` if desired.
* Use `--max-wait-hours` to tame extreme outliers before visualising.
