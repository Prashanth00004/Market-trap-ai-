import numpy as np
import pandas as pd
from datetime import datetime, timedelta


SAMPLE_HEADLINES = {
    "CRYPTO": [
        ("SEC unexpectedly delays spot ETF decision amid custody concerns", -0.85, "FLOW_REGULATORY"),
        ("Major exchange reports sudden outflows as whales move BTC to cold storage", 0.74, "FLOW_REGULATORY"),
        ("Surprise CPI print triggers immediate liquidation cascade across crypto", -0.80, "MACRO"),
        ("Institutional fund announces $500M Bitcoin treasury allocation at US open", 0.88, "FLOW_REGULATORY"),
        ("Perpetual funding rates spike to extreme levels ahead of daily close", -0.65, "FLOW_REGULATORY"),
    ],
    "GOLD": [
        ("Fed Chair signals rates will stay higher for longer after hot CPI data", -0.82, "MACRO"),
        ("Central banks accelerate gold reserves buying amid geopolitical escalation", 0.86, "MACRO"),
        ("US Non-Farm Payrolls smash estimates, sending DXY and 10Y yields surging", -0.79, "MACRO"),
        ("Emergency rate cut speculation lifts bullion past London session highs", 0.81, "MACRO"),
        ("Sudden Treasury yield reversal sparks sharp bullion short squeeze", 0.76, "MACRO"),
    ],
    "CRUDE": [
        ("EIA reports surprise 6.5M barrel crude inventory build, crushing demand outlook", -0.84, "INVENTORY_SUPPLY"),
        ("OPEC+ announces unexpected emergency production cut of 1.2M bpd", 0.90, "INVENTORY_SUPPLY"),
        ("Strait shipping disruption sparks immediate supply shock fears in WTI", 0.85, "INVENTORY_SUPPLY"),
        ("Global manufacturing PMI contracts for third straight month", -0.68, "MACRO"),
        ("API private data shows unexpected gasoline and distillate stock build", -0.72, "INVENTORY_SUPPLY"),
    ],
}

KILLZONE_HOURS = {
    "CRYPTO": {13, 14, 15, 23, 0, 1, 7, 8},
    "GOLD": {7, 8, 12, 13, 14, 15},
    "CRUDE": {13, 14, 15, 18},
}


def generate_multi_asset_dataset(n_bars: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic 5-minute OHLCV intraday bars for Crypto, Gold, and Crude Oil.
    Models:
      - Equal Highs / Equal Lows liquidity pools (BSL / SSL) built prior to stop runs
      - Macro News Sweeps, Session Killzone Judas Swings, and Liquidation Cascades
      - Genuine catalyst-backed Breakouts (negative class) for realistic ML discrimination
    """
    rng = np.random.default_rng(seed)
    start_time = datetime(2026, 1, 5, 0, 0)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n_bars)]

    base_prices = {"CRYPTO": 92000.0, "GOLD": 2740.0, "CRUDE": 74.50}
    volatilities = {"CRYPTO": 0.0020, "GOLD": 0.00085, "CRUDE": 0.0015}

    ohlcv_records = []
    news_records = []

    for asset, start_px in base_prices.items():
        vol = volatilities[asset]
        kz_hours = KILLZONE_HOURS[asset]
        price = start_px
        i = 0

        while i < n_bars:
            ts = timestamps[i]
            in_kz = ts.hour in kz_hours

            # Institutional sweeps cluster heavily during active Killzones
            event_prob = 0.048 if in_kz else 0.011

            if 35 < i < n_bars - 15 and rng.random() < event_prob:
                direction = 1 if rng.random() < 0.5 else -1  # +1 = upside BSL sweep, -1 = downside SSL sweep

                # Ensure pool_anchor sits strictly outside the last 18 bars' range so it forms a clean Equal High/Low pool
                recent_slice = ohlcv_records[-18:]
                if direction == 1:
                    recent_max = max(r["high"] for r in recent_slice)
                    pool_anchor = max(price * (1 + vol * 1.2), recent_max * (1 + vol * 0.35))
                else:
                    recent_min = min(r["low"] for r in recent_slice)
                    pool_anchor = min(price * (1 - vol * 1.2), recent_min * (1 - vol * 0.35))

                # Seed 2-3 Equal Highs or Equal Lows within tight tolerance over prior 4-12 bars
                for offset in [3, 7, 11]:
                    idx = len(ohlcv_records) - offset
                    if idx >= 0 and ohlcv_records[idx]["asset"] == asset:
                        if direction == 1:
                            ohlcv_records[idx]["high"] = pool_anchor * (1 + rng.uniform(-0.00015, 0.00015))
                        else:
                            ohlcv_records[idx]["low"] = pool_anchor * (1 + rng.uniform(-0.00015, 0.00015))

                # ~68% of extreme sweeps inside killzones trap late chasers vs ~42% off-hours
                trap_prob = 0.68 if in_kz else 0.42
                is_trap = rng.random() < trap_prob

                if is_trap:
                    if in_kz and rng.random() < 0.45:
                        catalyst_regime = "SESSION_KILLZONE_SWEEP"
                    elif asset == "CRYPTO" and rng.random() < 0.40:
                        catalyst_regime = "LEVERAGE_LIQUIDATION_SWEEP"
                    else:
                        catalyst_regime = "MACRO_NEWS_SWEEP"
                else:
                    catalyst_regime = "TRUE_BREAKOUT"

                # Attach breaking news headline (~85% of events have a breaking catalyst)
                has_news = (catalyst_regime != "SESSION_KILLZONE_SWEEP") or (rng.random() < 0.55)
                if has_news:
                    candidates = [h for h in SAMPLE_HEADLINES[asset] if (h[1] * direction) > 0]
                    headline, _, cat_tag = candidates[rng.integers(0, len(candidates))]
                    news_records.append({
                        "timestamp": ts,
                        "asset": asset,
                        "headline": headline,
                        "catalyst_tag": cat_tag,
                        "true_event_type": "TRAP" if is_trap else "BREAKOUT",
                        "catalyst_regime": catalyst_regime,
                        "direction": "BULL" if direction == 1 else "BEAR",
                    })

                # Impulse Candle (Bar i): Pierces pool_anchor on high ATR expansion & volume surge
                open_px = price
                pierce_dist = price * vol * rng.uniform(1.8, 3.0)
                target_pierce = pool_anchor + direction * pierce_dist

                if is_trap:
                    # Wick sweeps liquidity pool; close rejects back near/inside pool
                    close_px = pool_anchor - direction * price * vol * rng.uniform(0.15, 0.85)
                    if direction == 1:
                        high_px = target_pierce
                        low_px = min(open_px, close_px) - price * vol * 0.20
                    else:
                        low_px = target_pierce
                        high_px = max(open_px, close_px) + price * vol * 0.20
                else:
                    # Valid breakout closes strongly beyond the pool
                    close_px = target_pierce
                    high_px = max(open_px, close_px) + price * vol * rng.uniform(0.15, 0.35)
                    low_px = min(open_px, close_px) - price * vol * rng.uniform(0.15, 0.35)

                volume = rng.uniform(3200, 5600)
                ohlcv_records.append({
                    "timestamp": ts,
                    "asset": asset,
                    "open": open_px,
                    "high": high_px,
                    "low": low_px,
                    "close": close_px,
                    "volume": volume,
                    "event_label": 1 if is_trap else 0,
                    "catalyst_regime": catalyst_regime,
                })
                price = close_px
                i += 1

                # Next 3 bars: Market Structure Shift (MSS) displacement reversal for Traps vs. Continuation
                for _ in range(3):
                    if i >= n_bars:
                        break
                    step_ts = timestamps[i]
                    if is_trap:
                        # Killzone traps reverse cleanly; off-hours traps have noisier chop
                        rev_scale = rng.uniform(1.5, 2.5) if in_kz else rng.uniform(-0.3, 1.8)
                        rev_move = -direction * price * vol * rev_scale
                        o = price
                        c = o + rev_move
                        h = max(o, c) + abs(rev_move) * rng.uniform(0.1, 0.25)
                        l = min(o, c) - abs(rev_move) * rng.uniform(0.1, 0.25)
                        v = rng.uniform(2200, 3800)
                    else:
                        cont_move = direction * price * vol * rng.uniform(0.9, 1.8)
                        o = price
                        c = o + cont_move
                        h = max(o, c) + abs(cont_move) * rng.uniform(0.15, 0.3)
                        l = min(o, c) - abs(cont_move) * rng.uniform(0.15, 0.3)
                        v = rng.uniform(2100, 3600)

                    ohlcv_records.append({
                        "timestamp": step_ts,
                        "asset": asset,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": v,
                        "event_label": 0,
                        "catalyst_regime": "NONE",
                    })
                    price = c
                    i += 1
            else:
                # Normal intraday oscillation
                ret = rng.normal(0, vol * 0.85)
                o = price
                c = o * (1 + ret)
                spread = abs(c - o) + o * vol * rng.uniform(0.30, 0.80)
                h = max(o, c) + spread * rng.uniform(0.2, 0.5)
                l = min(o, c) - spread * rng.uniform(0.2, 0.5)
                v = rng.uniform(850, 1500)

                ohlcv_records.append({
                    "timestamp": ts,
                    "asset": asset,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "event_label": 0,
                    "catalyst_regime": "NONE",
                })
                price = c
                i += 1

    return pd.DataFrame(ohlcv_records), pd.DataFrame(news_records)
