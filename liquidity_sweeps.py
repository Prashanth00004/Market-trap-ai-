import numpy as np
import pandas as pd


class LiquiditySweepAnalyzer:
    """
    Models institutional liquidity sweep mechanics:
      1. Session & Killzone timing (London Open, NY Open / Macro, EIA Window, US ETF Open, Daily Reset)
      2. Buy-Side Liquidity (BSL) & Sell-Side Liquidity (SSL) pools via Equal Highs / Equal Lows
      3. Stop-run sweep detection + Market Structure Shift (MSS) displacement verification
    """

    def __init__(self, config: dict):
        self.config = config

    def encode_killzones(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        data = df.copy()
        ts = pd.to_datetime(data["timestamp"])
        hours = ts.dt.hour + ts.dt.minute / 60.0

        # Cyclical time-of-day encoding for ML generalization
        data["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
        data["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)

        killzones = self.config["assets"][asset].get("killzones_utc", [])
        in_any_kz = np.zeros(len(data), dtype=int)
        active_kz_name = np.array(["OFF_HOURS"] * len(data), dtype=object)

        for kz in killzones:
            start_h = kz["start_hour"]
            end_h = kz["end_hour"]
            h_int = ts.dt.hour
            if start_h <= end_h:
                mask = (h_int >= start_h) & (h_int < end_h)
            else:
                # Wraps midnight UTC (e.g., 23:00 to 02:00)
                mask = (h_int >= start_h) | (h_int < end_h)

            in_any_kz = np.where(mask, 1, in_any_kz)
            active_kz_name = np.where(mask, kz["name"], active_kz_name)

        data["in_active_killzone"] = in_any_kz
        data["active_killzone_name"] = active_kz_name
        return data

    def detect_liquidity_pools_and_sweeps(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        params = self.config["assets"][asset]
        lookback = params["lookback_range_bars"]
        tol = params.get("equal_level_tolerance_pct", 0.001)
        reentry_win = params["reentry_window_bars"]

        data = self.encode_killzones(df, asset)

        highs = data["high"].values
        lows = data["low"].values
        closes = data["close"].values
        n = len(data)

        bsl_pool_level = np.full(n, np.nan)
        ssl_pool_level = np.full(n, np.nan)
        bsl_touches = np.zeros(n, dtype=int)
        ssl_touches = np.zeros(n, dtype=int)

        for i in range(5, n):
            start_idx = max(0, i - lookback)
            win_highs = highs[start_idx:i]
            win_lows = lows[start_idx:i]

            max_h = np.max(win_highs)
            min_l = np.min(win_lows)

            # Count how many bars formed Equal Highs / Equal Lows near the range boundary
            equal_h_count = int(np.sum(np.abs(win_highs - max_h) / max_h <= tol))
            equal_l_count = int(np.sum(np.abs(win_lows - min_l) / min_l <= tol))

            bsl_pool_level[i] = max_h
            ssl_pool_level[i] = min_l
            bsl_touches[i] = equal_h_count
            ssl_touches[i] = equal_l_count

        data["bsl_pool_level"] = pd.Series(bsl_pool_level).bfill()
        data["ssl_pool_level"] = pd.Series(ssl_pool_level).bfill()
        data["bsl_pool_touches"] = bsl_touches
        data["ssl_pool_touches"] = ssl_touches

        # Distance from close to BSL / SSL pools (%)
        data["dist_to_bsl_pct"] = (data["bsl_pool_level"] - data["close"]) / data["close"]
        data["dist_to_ssl_pct"] = (data["close"] - data["ssl_pool_level"]) / data["close"]

        # Sweep triggers: Wick pierces BSL or SSL pool
        data["swept_bsl"] = (data["high"] > data["bsl_pool_level"]).astype(int)
        data["swept_ssl"] = (data["low"] < data["ssl_pool_level"]).astype(int)

        # Market Structure Shift (MSS) confirmation: price closes back inside pool within reentry_win
        future_min_close = (
            data["close"].shift(-1).rolling(reentry_win, min_periods=1).min().shift(-(reentry_win - 1))
        )
        future_max_close = (
            data["close"].shift(-1).rolling(reentry_win, min_periods=1).max().shift(-(reentry_win - 1))
        )

        bsl_mss = (data["swept_bsl"] == 1) & (
            (data["close"] < data["bsl_pool_level"]) | (future_min_close < data["bsl_pool_level"])
        )
        ssl_mss = (data["swept_ssl"] == 1) & (
            (data["close"] > data["ssl_pool_level"]) | (future_max_close > data["ssl_pool_level"])
        )

        data["mss_confirmed"] = (bsl_mss | ssl_mss).astype(int)
        data["sweep_classification"] = np.where(
            bsl_mss,
            "BSL_BULL_TRAP_SWEEP",
            np.where(
                ssl_mss,
                "SSL_BEAR_TRAP_SWEEP",
                np.where((data["swept_bsl"] == 1) | (data["swept_ssl"] == 1), "UNCONFIRMED_SWEEP", "NONE"),
            ),
        )
        return data
