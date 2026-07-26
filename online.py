"""Pure online next-return prediction with River.

Every price column is both an input and a prediction target. CSV mode replays rows
immediately. Live mode blocks until the next row arrives. Results are always appended
to predictions.csv.
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
    """Return the percentage movement from previous to current."""
    if previous == 0:
        raise ValueError("A previous price is zero, so percentage change is undefined")
    return (current / previous - 1.0) * 100.0


def price_changes(previous: Row, current: Row, date_column: str) -> dict[str, float]:
    """Use every asset's latest percentage movement as a feature."""
    return {
        name: percentage_change(float(previous[name]), float(value))
        for name, value in current.items()
        if name != date_column
    }


def make_model():
    """Small online linear model updated one row at a time."""
    return preprocessing.StandardScaler() | linear_model.LinearRegression()


def csv_rows(path: Path) -> Iterator[Row]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        yield from reader


def live_rows(columns: list[str]) -> Iterator[Row]:
    """Block until a comma-separated price row arrives on stdin."""
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


def run(rows: Iterable[Row], date_column: str) -> None:
    """Predict all assets, reveal results, log them, and learn online."""
    models: dict[str, object] = {}
    previous: Row | None = None
    pending_x: dict[str, float] | None = None
    pending_predictions: dict[str, float] = {}
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

            if previous is None:
                previous = current
                continue

            current_x = price_changes(previous, current, date_column)

            if not models:
                models = {target: make_model() for target in current_x}

            if set(current_x) != set(models):
                raise ValueError("Price columns changed between rows")

            # The predictions made from the previous feature row are now resolved.
            if pending_x is not None:
                for target, model in models.items():
                    actual = current_x[target]
                    prediction = pending_predictions[target]
                    writer.writerow(
                        {
                            "prediction_date": pending_date,
                            "result_date": current[date_column],
                            "target": target,
                            "prediction_percent": prediction,
                            "actual_percent": actual,
                            "error_percent": actual - prediction,
                        }
                    )
                    model.learn_one(pending_x, actual)

                log_file.flush()

            # The current cross-market movements predict every asset's next movement.
            pending_x = current_x
            pending_predictions = {
                target: model.predict_one(current_x)
                for target, model in models.items()
            }
            pending_date = current[date_column]
            previous = current

    if pending_date is not None:
        print(
            f"Predictions from {pending_date} are waiting for the next row. "
            f"Resolved predictions were logged to {LOG_PATH}.",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("csv", "live"), default="csv")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--csv", type=Path, help="Historical price CSV for CSV mode")
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Live row columns, including date and all asset prices",
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

    run(rows, date_column=args.date_column)


if __name__ == "__main__":
    main()
