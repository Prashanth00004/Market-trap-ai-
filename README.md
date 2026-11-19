# Market Trap AI — Institutional Liquidity Sweep, Sentiment & Prediction Engine

An event-driven quantitative intelligence and machine learning system designed to detect **false breakouts, breakdowns, liquidity sweeps, and reversals (Market Traps)** across **Crypto (`BTC-USD`), Gold (`XAU-USD`), and Crude Oil (`WTI-OIL`)** by fusing breaking news sentiment with institutional order-flow microstructure.

---

## Architectural Highlights

1. **Institutional Liquidity Sweep Engine (`src/liquidity_sweeps.py`)**
   - **Buy-Side & Sell-Side Liquidity Pools (`BSL` / `SSL`):** Identifies clustered Equal Highs and Equal Lows where resting stop-losses and breakout orders accumulate.
   - **Session & Killzone Timing (`UTC`):** Encodes cyclical time-of-day features alongside asset-specific institutional windows (London Open Judas Swings, NY Open & US Macro releases, London PM Fix, Wednesday EIA Crude Inventory windows, US Spot ETF Open, and 00:00 UTC Funding Resets).
   - **Market Structure Shift (`MSS`) Confirmation:** Separates genuine liquidity sweeps (piercing a BSL/SSL pool followed by displacement back inside the range) from valid trend breakouts.

2. **Breaking News & Catalyst Sentiment Engine (`src/sentiment_engine.py`)**
   - Routes headlines to `CRYPTO`, `GOLD`, or `CRUDE`, classifies catalyst regimes (`MACRO`, `INVENTORY_SUPPLY`, `FLOW_REGULATORY`), and computes sentiment (`[-1.0, +1.0]`) and urgency scores (`[0.0, 1.0]`). Includes optional drop-in `ProsusAI/finbert` transformer support.

3. **Hybrid Ensemble Predictor (`src/predictor.py`)**
   - Combines NLP sentiment-price divergence with sweep geometry (`bsl_pool_touches`, `ssl_pool_touches`, `in_active_killzone`, `atr_ratio`, `volume_ratio`, `wick_ratios`, `vwap_dist_pct`, `rsi_14`) via a calibrated `GradientBoosting + RandomForest` soft-voting ensemble.

4. **Cross-Asset & Killzone Analytics (`src/analytics.py`)**
   - Evaluates Win Rate, Profit Factor, Sharpe Ratio, Max Drawdown, Killzone vs. Off-Hours edge attribution, Catalyst Regime attribution, and cross-asset return correlations.

5. **Real-Time Inference API (`src/api.py`)**
   - FastAPI service providing `/analyze/sentiment`, `/predict/trap`, `/analytics/evaluation`, and `/analytics/killzones`.

---

## Project Structure

```text
market_trap_ai/
├── README.md                     # Documentation & architecture guide
├── requirements.txt              # Python dependencies
├── config.json                   # Asset thresholds, tolerances & UTC killzones
├── main.py                       # Full evaluation, backtest & report generator
├── src/
│   ├── __init__.py
│   ├── data_generator.py         # Multi-asset OHLCV + liquidity pool + news simulator
│   ├── sentiment_engine.py       # Financial sentiment, urgency & catalyst router
│   ├── liquidity_sweeps.py       # BSL/SSL liquidity pools, UTC killzones & MSS verifier
│   ├── trap_detector.py          # Unified microstructure & trap detection pipeline
│   ├── predictor.py              # Hybrid ML ensemble classifier & trade action engine
│   ├── analytics.py              # Backtest metrics, killzone attribution & correlations
│   └── api.py                    # FastAPI REST inference & analytics endpoints
├── data/
│   ├── evaluation_report.json    # Complete ML & strategy performance metrics
│   ├── evaluated_signals.csv     # Full multi-asset feature & prediction dataset
│   └── detected_traps_log.csv    # Filtered log of all detected sweeps & trade outcomes
└── tests/
    └── test_pipeline.py          # Automated end-to-end unit & API tests
```

---

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run End-to-End Evaluation & Backtest**
   ```bash
   python main.py
   ```

3. **Run Automated Test Suite**
   ```bash
   python -m unittest discover -s tests
   ```

4. **Launch Real-Time FastAPI Server**
   ```bash
   uvicorn src.api:app --reload --port 8000
   ```
