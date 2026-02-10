# Quickstart

## Run the UI in one command

From the repository root:

```bash
npm run dev
```

Then open:

`http://localhost:8501`

The launcher will automatically:
- create `.venv` if needed,
- install dependencies,
- start Streamlit.

## First actions in the app

1. Enable `Fast mode`.
2. Keep `Use cached example dataset`.
3. Click `Load data`.
4. Click `Run analysis`.

## Use real market data

In `Data Source` choose `Fetch with yfinance`, set dates/tickers, then click:
1. `Load data`
2. `Run analysis`

## Generate videos

In sidebar `Animation / Export`:
1. Select one or more animations.
2. Click `Generate animations`.
3. Open `Animations / Export` tab to preview/download.

Generated files are saved to:

`runs/<timestamp>/videos/`
