# River online price prediction

A minimal online linear-regression experiment. There is no train/test split.
Every row is handled in chronological order:

1. A new price row arrives.
2. The previous prediction is compared with the newly revealed target return.
3. The model learns from that one example with `learn_one`.
4. Current percentage movements are used to predict the target's next return.
5. The final prediction remains pending until another row arrives.

The percentage-change formula is:

```text
(current_price / previous_price - 1) * 100
```

## Install

```bash
pip install -r requirements.txt
```

## CSV mode

The CSV must contain prices, not precomputed returns, and must already be sorted oldest to newest.

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

The program writes simple rows to stdout:

```csv
date,prediction_percent,actual_percent,error_percent
```

Save them when desired:

```bash
python online.py --mode csv --csv markets.csv --target EURUSD > predictions.csv
```

## Live mode

Live mode uses the exact same learning loop and blocks until you enter the next row. This stdin interface is deliberately simple and can later be replaced by an API stream without changing `run`.

```bash
python online.py --mode live \
  --target EURUSD \
  --columns date EURUSD GBPUSD USDJPY GOLD OIL_WTI
```

Then enter one row whenever prices update:

```text
2026-07-26T12:00:00,1.1700,1.3400,154.20,2400.0,78.0
2026-07-26T13:00:00,1.1710,1.3390,154.10,2403.0,78.4
```

The first row establishes prices. The second creates the first feature vector and prediction. The third row reveals whether that prediction was right, updates the model, and produces the next prediction.

This is research code, not a trading system or market-data feed.
