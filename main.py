import json
import pandas as pd
from src.data_generator import generate_multi_asset_dataset
from src.sentiment_engine import FinancialSentimentEngine
from src.trap_detector import MarketTrapDetector
from src.predictor import MarketTrapPredictor
from src.analytics import CrossAssetAnalytics


def run_evaluation():
    print("=" * 72)
    print("  MARKET TRAP AI (v2.0) — LIQUIDITY SWEEP & SENTIMENT EVALUATION")
    print("=" * 72)

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # 1. Generate Multi-Asset Data (Crypto, Gold, Crude Oil)
    print("\n[1/5] Generating multi-asset OHLCV, liquidity pools & breaking news...")
    ohlcv_df, news_df = generate_multi_asset_dataset(n_bars=1800, seed=42)
    print(f"      Loaded {len(ohlcv_df)} OHLCV bars and {len(news_df)} breaking news catalysts.")

    # 2. NLP Sentiment Scoring & Asset Routing
    print("[2/5] Running Financial Sentiment, Catalyst & Urgency Scoring...")
    sentiment_engine = FinancialSentimentEngine()
    enriched_news = sentiment_engine.enrich_news_dataframe(news_df)

    # 3. Institutional Liquidity Sweep & Market Trap Detection
    print("[3/5] Detecting BSL/SSL Liquidity Pools, Killzones & MSS Reversals...")
    detector = MarketTrapDetector(config)
    asset_frames = []
    for asset in ["CRYPTO", "GOLD", "CRUDE"]:
        sub = ohlcv_df[ohlcv_df["asset"] == asset]
        asset_frames.append(detector.compute_features(sub, asset))
    market_features = pd.concat(asset_frames, ignore_index=True)

    # 4. Hybrid ML Ensemble Training & Evaluation
    print("[4/5] Training & Evaluating Hybrid Sentiment-Sweep Ensemble Predictor...")
    predictor = MarketTrapPredictor(random_state=config["model"]["random_state"])
    fused_df = predictor.fuse_news_and_market(market_features, enriched_news)
    ml_metrics = predictor.train_and_evaluate(fused_df, test_size=config["model"]["test_size"])

    # 5. Cross-Asset & Killzone Strategy Analytics
    print("[5/5] Computing Cross-Asset, Killzone & Catalyst Attribution Analytics...")
    analytics_report = CrossAssetAnalytics.evaluate_trap_strategy(fused_df)

    full_report = {
        "project": "Market Trap AI v2.0 (Institutional Liquidity Sweep Edition)",
        "assets_covered": ["CRYPTO (BTC-USD)", "GOLD (XAU-USD)", "CRUDE (WTI-OIL)"],
        "ml_classifier_evaluation": ml_metrics,
        "strategy_and_cross_asset_analytics": analytics_report,
    }

    with open("data/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    fused_df.to_csv("data/evaluated_signals.csv", index=False)

    traps_log = fused_df[fused_df["is_detected_trap"] == 1][
        [
            "timestamp",
            "asset",
            "active_killzone_name",
            "detected_trap_type",
            "catalyst_regime",
            "bsl_pool_touches",
            "ssl_pool_touches",
            "atr_ratio",
            "volume_ratio",
            "sentiment_score",
            "sentiment_price_divergence",
            "headline",
            "fwd_3bar_return",
        ]
    ]
    traps_log.to_csv("data/detected_traps_log.csv", index=False)

    print("\n" + "=" * 72)
    print("  EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  Ensemble ROC-AUC       : {ml_metrics['roc_auc']}")
    print(f"  Precision / Recall     : {ml_metrics['precision']} / {ml_metrics['recall']} (F1: {ml_metrics['f1_score']})")
    print("-" * 72)
    for asset, stats in analytics_report["by_asset"].items():
        print(
            f"  {asset:<7} | Traps: {stats['detected_traps']:<3} "
            f"| Win Rate: {stats['win_rate_pct']:>5.1f}% "
            f"| Profit Factor: {stats['profit_factor']:>5.2f} "
            f"| Sharpe: {stats['sharpe_ratio']:>5.2f}"
        )
    print("-" * 72)
    kz_stats = analytics_report["killzone_attribution"]["in_active_killzone"]
    off_stats = analytics_report["killzone_attribution"]["off_hours"]
    print(
        f"  KILLZONES | Traps: {kz_stats['detected_traps']:<3} | Win Rate: {kz_stats['win_rate_pct']:>5.1f}% "
        f"| Avg Return: {kz_stats['avg_return_bps']:>6.1f} bps"
    )
    print(
        f"  OFF-HOURS | Traps: {off_stats['detected_traps']:<3} | Win Rate: {off_stats['win_rate_pct']:>5.1f}% "
        f"| Avg Return: {off_stats['avg_return_bps']:>6.1f} bps"
    )
    print("=" * 72)
    print("Saved JSON report      -> data/evaluation_report.json")
    print("Saved full signals     -> data/evaluated_signals.csv")
    print("Saved traps audit log  -> data/detected_traps_log.csv")


if __name__ == "__main__":
    run_evaluation()
