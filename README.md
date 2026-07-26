# River online price prediction

A minimal online linear-regression experiment with no train/test split.
All price columns are used as percentage-change inputs, while one target is selected with `--target`.

For every new row:

1. The previous prediction is compared with the newly revealed target return.
2. The result is appended to `predictions.csv`.
3. The model learns from that one example with `learn_one`.
4. Current percentage movements are used to predict the target's next return.
5. The latest prediction remains pending until another row arrives.

The percentage-change formula is:

```text
(current_price / previous_price - 1) * 100
```

## Install

```bash
pip install -r requirements.txt
```

## CSV mode

The CSV must contain prices, not precomputed returns, and must be sorted oldest to newest.

```csv
date,EURUSD,GBPUSD,USDJPY,GOLD,OIL_WTI
2024-01-02,1.0942,1.2618,142.10,2073.4,70.38
2024-01-03,1.0921,1.2663,143.28,2042.8,72.70
2024-01-04,1.0950,1.2680,144.10,2050.0,72.20
```

Run:

```bash
python online.py --mode csv --csv markets.csv --target EURUSD
```

Results are always appended to `predictions.csv`.

## Live mode

```bash
python online.py --mode live \
  --target EURUSD \
  --columns date EURUSD GBPUSD USDJPY GOLD OIL_WTI
```

The first row establishes prices. The second creates percentage-change features and the first prediction. The third resolves that prediction, logs it, updates the model, and creates the next prediction.

This is research code, not a trading system or market-data feed.
