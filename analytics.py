import numpy as np
import pandas as pd


class CrossAssetAnalytics:
    """
    Evaluates trap-fading strategy performance across Crypto, Gold, and Crude Oil,
    including Killzone vs. Off-Hours edge attribution, Catalyst Regime breakdown,
    and Cross-Asset return correlations.
    """

    @staticmethod
    def _compute_trade_stats(traps_df: pd.DataFrame) -> dict:
        if traps_df.empty:
            return {
                "detected_traps": 0,
                "win_rate_pct": 0.0,
                "avg_return_bps": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
            }

        trade_dir = np.where(traps_df["detected_trap_type"] == "BULL_TRAP", -1.0, 1.0)
        strat_returns = trade_dir * traps_df["fwd_3bar_return"]

        wins = int((strat_returns > 0).sum())
        total = int(len(strat_returns))
        win_rate = (wins / total) * 100.0 if total > 0 else 0.0

        gross_profit = float(strat_returns[strat_returns > 0].sum())
        gross_loss = float(abs(strat_returns[strat_returns < 0].sum()))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.99

        cum_curve = (1 + strat_returns).cumprod()
        peak = cum_curve.cummax()
        max_dd = float(((cum_curve - peak) / peak).min() * 100.0)

        sharpe = (
            float((strat_returns.mean() / strat_returns.std()) * np.sqrt(252))
            if strat_returns.std() > 0 else 0.0
        )

        return {
            "detected_traps": total,
            "bull_traps": int((traps_df["detected_trap_type"] == "BULL_TRAP").sum()),
            "bear_traps": int((traps_df["detected_trap_type"] == "BEAR_TRAP").sum()),
            "win_rate_pct": round(win_rate, 2),
            "avg_return_bps": round(float(strat_returns.mean() * 10000), 2),
            "profit_factor": profit_factor,
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
        }

    @classmethod
    def evaluate_trap_strategy(cls, fused_df: pd.DataFrame) -> dict:
        asset_metrics = {}
        for asset, group in fused_df.groupby("asset"):
            traps = group[group["is_detected_trap"] == 1].copy()
            asset_metrics[asset] = cls._compute_trade_stats(traps)

        all_traps = fused_df[fused_df["is_detected_trap"] == 1].copy()

        # Killzone vs. Off-Hours Attribution
        kz_traps = all_traps[all_traps["in_active_killzone"] == 1]
        off_traps = all_traps[all_traps["in_active_killzone"] == 0]
        killzone_attribution = {
            "in_active_killzone": cls._compute_trade_stats(kz_traps),
            "off_hours": cls._compute_trade_stats(off_traps),
        }

        # Catalyst Regime Breakdown
        regime_breakdown = {}
        if "catalyst_regime" in all_traps.columns:
            for regime, r_group in all_traps.groupby("catalyst_regime"):
                if regime != "NONE":
                    regime_breakdown[regime] = cls._compute_trade_stats(r_group)

        # Cross-asset return correlation matrix
        pivot_close = fused_df.pivot(index="timestamp", columns="asset", values="close").pct_change().dropna()
        corr_matrix = pivot_close.corr().round(4).to_dict()

        return {
            "by_asset": asset_metrics,
            "killzone_attribution": killzone_attribution,
            "by_catalyst_regime": regime_breakdown,
            "total_traps_evaluated": int(len(all_traps)),
            "cross_asset_return_correlation": corr_matrix,
        }
