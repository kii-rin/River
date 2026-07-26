"""Pure online next-return prediction with River.

All price columns are used as percentage-change inputs. One target column is selected
with --target. CSV mode replays rows immediately; live mode waits for the next row.
Resolved predictions are always appended to predictions.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from river import linear_model, preprocessing


Row = dict[str, str]
LOG_PATH = Path("predictions.csv")


def percentage_change(previous: float, current: float) -> float:
    if previous == 0:
        raise ValueError("A previous price is zero, so percentage change is undefined")
    return (current / previous - 1.0) * 100.0


def price_changes(previous: Row, current: Row, date_column: str) -> dict[str, float]:
    return {
        name: percentage_change(float(previous[name]), float(value))
        for name, value in current.items()
        if name != date_column
    }


def make_model():
    return preprocessing.StandardScaler() | linear_model.LinearRegression()


def csv_rows(path: Path) -> Iterator[Row]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        yield from reader


def live_rows(columns: list[str]) -> Iterator[Row]:
    print("Live columns:", ",".join(columns), file=sys.stderr)
    print("Enter one comma-separated row at a time. Ctrl-D/Ctrl-Z stops.", file=sys.stderr)

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            return

        if not line:
            continue

        values = next(csv.reader([line]))
        if len(values) != len(columns):
            print(f"Expected {len(columns)} values, received {len(values)}", file=sys.stderr)
            continue

        yield dict(zip(columns, values, strict=True))


def run(rows: Iterable[Row], target: str, date_column: str) -> None:
    model = make_model()
    previous: Row | None = None
    pending_x: dict[str, float] | None = None
    pending_prediction: float | None = None
    pending_date: str | None = None

    log_exists = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0

    with LOG_PATH.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "prediction_date",
                "result_date",
                "target",
                "prediction_percent",
                "actual_percent",
                "error_percent",
            ],
        )
        if not log_exists:
            writer.writeheader()

        for current in rows:
            if date_column not in current:
                raise ValueError(f"Date column {date_column!r} was not found")
            if target not in current:
                raise ValueError(f"Target column {target!r} was not found")

            if previous is None:
                previous = current
                continue

            current_x = price_changes(previous, current, date_column)

            if pending_x is not None and pending_prediction is not None:
                actual = current_x[target]
                writer.writerow(
                    {
                        "prediction_date": pending_date,
                        "result_date": current[date_column],
                        "target": target,
                        "prediction_percent": pending_prediction,
                        "actual_percent": actual,
                        "error_percent": actual - pending_prediction,
                    }
                )
                log_file.flush()
                model.learn_one(pending_x, actual)

            pending_x = current_x
            pending_prediction = model.predict_one(current_x)
            pending_date = current[date_column]
            previous = current

    if pending_date is not None:
        print(
            f"Prediction from {pending_date} is waiting for the next row. "
            f"Resolved predictions were logged to {LOG_PATH}.",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("csv", "live"), default="csv")
    parser.add_argument("--target", required=True)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--csv", type=Path, help="Historical price CSV for CSV mode")
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Live row columns, including date, target, and input prices",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "csv":
        if args.csv is None:
            raise SystemExit("CSV mode requires --csv")
        rows = csv_rows(args.csv)
    else:
        if not args.columns:
            raise SystemExit("Live mode requires --columns")
        rows = live_rows(args.columns)

    run(rows, target=args.target, date_column=args.date_column)


if __name__ == "__main__":
    main()
