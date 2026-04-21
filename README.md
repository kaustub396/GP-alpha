# GP-alpha: Regime-Aware Formulaic Alpha Discovery (NIFTY-50)

This project builds and evaluates **formulaic alpha strategies** for Indian equities, with a focus on **regime-aware modeling** (Bull/Bear market states) using Genetic Programming (GP).

## What this project does

1. Downloads and standardizes historical market data for NIFTY-50 constituents.
2. Builds cross-sectional technical features from OHLCV data.
3. Detects market regimes (Bull/Bear) using an HMM-based detector.
4. Trains GP-based alpha formulas (vanilla and regime-aware variants).
5. Backtests strategies and reports IC/Rank-IC, ICIR, return, Sharpe, drawdown, turnover, and win rate.

## Dataset used

- **Universe**: NIFTY-50 stocks + NIFTY50 index proxy
- **Raw files**: `data/raw/*.csv`
- **Standardized files**: `data/processed/*.csv`
- **Feature files**: `data/processed/features/*.csv`
- **Panel object**: `data/processed/panel.pkl`

### Coverage (from `data/data_summary.csv`)

- ~51 instruments (NIFTY-50 constituents + index)
- Earliest dates start around **2000-01-03** for many symbols
- Latest dates go up to **2025-12-30** for most symbols
- No dropped rows reported in the summary file

## Repository structure

- `download_data.py` - data download pipeline
- `standardize_data.py` - schema cleanup/normalization
- `build_features.py` - feature engineering and panel construction
- `regime_detector.py` - HMM regime classification
- `gp_engine.py` - GP engine and formula search
- `final_experiments.py` - final experiment runner
- `evaluation.py` - backtest and metric computation
- `baselines.py` - baseline strategy implementations
- `data/` - datasets and result snapshots
- `figures/` - plots/tables used for analysis

## Results obtained

The repository currently contains multiple result snapshots.  
Primary files:

- `data/main_results.csv`
- `data/main_results_v2.csv`
- `data/baseline_results.csv`

### Snapshot: `data/main_results.csv`

| Strategy | Ann Return (Net) | Sharpe (Net) | Max Drawdown |
|---|---:|---:|---:|
| Regime-Aware GP (Ours) | 2.81% | 0.1765 | -37.29% |
| Vanilla GP | 5.15% | 0.3112 | -20.56% |
| Mean Reversion | 4.46% | 0.2981 | -20.19% |
| Momentum (12-1M) | -2.57% | -0.1369 | -40.65% |

### Snapshot: `data/main_results_v2.csv`

| Strategy | Ann Return (Net) | Sharpe (Net) | Max Drawdown |
|---|---:|---:|---:|
| Regime-GP Soft (Ours) | -5.81% | -0.3888 | -49.04% |
| Regime-GP Hard (Ours) | -2.56% | -0.1924 | -35.34% |
| Vanilla GP | 15.77% | 1.0804 | -10.19% |
| Mean Reversion | 4.46% | 0.2981 | -20.19% |

For interpretation notes, see: `results_explanation.md`.

## How to run

```bash
python download_data.py
python standardize_data.py
python build_features.py
python final_experiments.py
```

Then inspect outputs in:

- `data/*.csv`
- `data/*.pkl`
- `figures/*.png`

## Notes

- Some large artifacts (e.g., `.pkl`) are committed directly; consider Git LFS if repository size grows.
- This is a research/backtesting project and not financial advice.
