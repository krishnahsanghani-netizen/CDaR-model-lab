"""Minimal end-to-end enhanced CDaR pipeline example."""

from enhanced_cdar.data.loaders import load_from_yfinance
from enhanced_cdar.data.preprocess import align_and_clean_prices
from enhanced_cdar.metrics.drawdown import compute_returns
from enhanced_cdar.portfolio.optimization import optimize_portfolio_cdar


def main() -> None:
    result = load_from_yfinance(
        tickers=["SPY", "AGG", "GLD", "QQQ"],
        start="2021-01-01",
        end="2026-01-01",
    )
    prices = align_and_clean_prices(result.prices)
    returns = compute_returns(prices)

    opt = optimize_portfolio_cdar(
        returns=returns,
        alpha=0.95,
        no_short=True,
    )
    print("status:", opt["status"])
    print("cdar:", opt["cdar"])
    print("weights:", opt["weights"])


if __name__ == "__main__":
    main()
