# CLI Usage Examples

## Fetch Data
```bash
enhanced-cdar fetch-data \
  --tickers SPY,AGG,GLD,QQQ \
  --start 2021-01-01 \
  --end 2026-01-01 \
  --output data/prices.csv
```

## Analyze Fixed Weights
```bash
enhanced-cdar analyze-portfolio \
  --prices-csv data/prices.csv \
  --weights 0.25,0.25,0.25,0.25 \
  --alpha 0.95 \
  --format text
```

## Optimize CDaR
```bash
enhanced-cdar optimize-cdar \
  --prices-csv data/prices.csv \
  --alpha 0.95 \
  --no-short \
  --format json
```

## Backtest with Rebalancing
```bash
enhanced-cdar backtest \
  --prices-csv data/prices.csv \
  --weights 0.25,0.25,0.25,0.25 \
  --rebalance-calendar M \
  --rebalance-every-n-periods 21 \
  --format json
```

## Build Frontier + Plot
```bash
enhanced-cdar frontier \
  --prices-csv data/prices.csv \
  --alpha 0.95 \
  --n-points 20 \
  --allow-short \
  --gross-limit 2.0 \
  --output-csv runs/frontier.csv \
  --plot-path runs/frontier.html
```

## Surface with Preset Grid
```bash
enhanced-cdar surface \
  --prices-csv data/prices.csv \
  --alpha 0.95 \
  --lambda-preset medium \
  --output-csv runs/surface.csv \
  --plot-path runs/surface_3d.html
```

## End-to-End Pipeline
```bash
enhanced-cdar run-pipeline
```
