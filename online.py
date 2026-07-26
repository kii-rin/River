"""Pure online next-return prediction with River.

CSV mode replays rows immediately. Live mode blocks on stdin until the next row
arrives. In both modes the order is always: reveal result -> learn -> predict next.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from river import linear_model, preprocessing


Row = dict[str, str]


def percentage_change(previous: float, current: float) -> float:
    """Return the percentage movement from previous to current."""
    if previous == 0:
        raise ValueError("A previous price is zero, so percentage change is undefined")
    return (current / previous - 1.0) * 100.0


def price_changes(previous: Row, current: Row, date_column: str) -> dict[str, float]:
    """Use each asset's latest percentage movement as a model feature."""
    features: dict[str, float] = {}

    for name, current_value in current.items():
        if name == date_column:
            continue
        if name not in previous:
            raise ValueError(f"Column {name!r} is missing from the previous row")

        features[name] = percentage_change(
            float(previous[name]),
            float(current_value),
        )

    return features


def make_model():
    """Small online linear model; both scaling and regression update one row at a time."""
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


def run(rows: Iterable[Row], target: str, date_column: str) -> None:
    """Predict before learning, then update only when the next result is known."""
    model = make_model()
    previous: Row | None = None
    pending_x: dict[str, float] | None = None
    pending_prediction: float | None = None
    pending_date: str | None = None

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["date", "prediction_percent", "actual_percent", "error_percent"],
    )
    writer.writeheader()

    for current in rows:
        if target not in current:
            raise ValueError(f"Target column {target!r} was not found")
        if date_column not in current:
            raise ValueError(f"Date column {date_column!r} was not found")

        if previous is None:
            previous = current
            continue

        current_x = price_changes(previous, current, date_column)
        actual_return = percentage_change(
            float(previous[target]),
            float(current[target]),
        )

        # The previous feature row predicted the return that has just arrived.
        if pending_x is not None and pending_prediction is not None:
            writer.writerow(
                {
                    "date": current[date_column],
                    "prediction_percent": pending_prediction,
                    "actual_percent": actual_return,
                    "error_percent": actual_return - pending_prediction,
                }
            )
            sys.stdout.flush()
            model.learn_one(pending_x, actual_return)

        # Today's cross-market movements predict the target's next movement.
        pending_x = current_x
        pending_prediction = model.predict_one(current_x)
        pending_date = current[date_column]
        previous = current

    if pending_date is not None:
        print(
            f"Waiting for the next row to reveal the result of the prediction made at {pending_date}.",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("csv", "live"), default="csv")
    parser.add_argument("--target", default="EURUSD")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--csv", type=Path, help="Historical price CSV for CSV mode")
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Live row columns, including date and target",
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
