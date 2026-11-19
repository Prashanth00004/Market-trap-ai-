import numpy as np
import pandas as pd
from src.liquidity_sweeps import LiquiditySweepAnalyzer


class MarketTrapDetector:
    """
    Computes price-action microstructure features, intraday VWAP deviation, and
    institutional liquidity sweeps (Bull Traps, Bear Traps, and Liquidity Grabs).
    """

    def __init__(self, config: dict):
        self.config = config
        self.sweep_analyzer = LiquiditySweepAnalyzer(config)

    def compute_features(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        params = self.config["assets"][asset]
        data = df.copy().sort_values("timestamp").reset_index(drop=True)

        # 1. Liquidity Pools, Killzones & Sweep Mechanics
        data = self.sweep_analyzer.detect_liquidity_pools_and_sweeps(data, asset)

        # 2. True Range & ATR (14)
        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        data["atr_14"] = tr.rolling(14, min_periods=1).mean()

        # 3. Candle Geometry & Wick Rejection Ratios
        candle_range = (data["high"] - data["low"]).replace(0, np.nan)
        body = (data["close"] - data["open"]).abs()
        upper_wick = data["high"] - data[["open", "close"]].max(axis=1)
        lower_wick = data[["open", "close"]].min(axis=1) - data["low"]

        data["atr_ratio"] = candle_range / data["atr_14"].replace(0, np.nan)
        data["body_ratio"] = (body / candle_range).fillna(0)
        data["upper_wick_ratio"] = (upper_wick / candle_range).fillna(0)
        data["lower_wick_ratio"] = (lower_wick / candle_range).fillna(0)

        # 4. Volume Surge & Rolling Intraday VWAP Deviation
        vol_ma = data["volume"].rolling(20, min_periods=1).mean()
        data["volume_ratio"] = data["volume"] / vol_ma.replace(0, np.nan)

        typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
        rolling_pv = (typical_price * data["volume"]).rolling(48, min_periods=1).sum()
        rolling_vol = data["volume"].rolling(48, min_periods=1).sum().replace(0, np.nan)
        data["rolling_vwap"] = rolling_pv / rolling_vol
        data["vwap_dist_pct"] = (data["close"] - data["rolling_vwap"]) / data["rolling_vwap"]

        # 5. RSI (14)
        delta = data["close"].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        data["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

        # 6. Combined Algorithmic Trap Detection (Liquidity Sweep + Volatility/Volume Impulse + MSS)
        impulse_ok = (
            (data["atr_ratio"] >= params["atr_impulse_multiplier"] * 0.70)
            & (data["volume_ratio"] >= params["volume_surge_multiplier"] * 0.80)
        )

        bull_trap_cond = (data["sweep_classification"] == "BSL_BULL_TRAP_SWEEP") & impulse_ok
        bear_trap_cond = (data["sweep_classification"] == "SSL_BEAR_TRAP_SWEEP") & impulse_ok

        data["detected_trap_type"] = np.where(
            bull_trap_cond,
            "BULL_TRAP",
            np.where(bear_trap_cond, "BEAR_TRAP", "NONE"),
        )
        data["is_detected_trap"] = (data["detected_trap_type"] != "NONE").astype(int)

        # Forward 3-bar return for backtesting trap fade trades
        data["fwd_3bar_return"] = (data["close"].shift(-3) - data["close"]) / data["close"]
        return data.fillna(0)
