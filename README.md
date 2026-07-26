# River online price prediction

A minimal online linear-regression experiment with no train/test split.
Every numeric price column is automatically used in two ways:

- as an input feature
- as its own prediction target

A separate River model is maintained for each asset.

For every new row:

1. The previous predictions are compared with the newly revealed returns.
2. All resolved predictions are appended to `predictions.csv`.
3. Each target model learns from that one example with `learn_one`.
4. Current percentage movements are used to predict every asset's next return.
5. The latest predictions remain pending until another row arrives.

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
python online.py --mode csv --csv markets.csv
```

Results are always appended to:

```text
predictions.csv
```

The log contains one row per target per resolved timestamp:

```csv
prediction_date,result_date,target,prediction_percent,actual_percent,error_percent
```

## Live mode

Live mode uses the same learning loop and waits until the next row arrives.
The stdin source can later be replaced by an API stream without changing `run`.

```bash
python online.py --mode live \
  --columns date EURUSD GBPUSD USDJPY GOLD OIL_WTI
```

Then enter one row whenever prices update:

```text
2026-07-26T12:00:00,1.1700,1.3400,154.20,2400.0,78.0
2026-07-26T13:00:00,1.1710,1.3390,154.10,2403.0,78.4
```

The first row establishes prices. The second creates percentage-change features and predictions for every asset. The third resolves those predictions, logs them, updates every model, and creates the next predictions.

This is research code, not a trading system or market-data feed.
