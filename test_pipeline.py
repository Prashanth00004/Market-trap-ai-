import json
import unittest
from src.data_generator import generate_multi_asset_dataset
from src.sentiment_engine import FinancialSentimentEngine
from src.trap_detector import MarketTrapDetector
from src.predictor import MarketTrapPredictor
from src.analytics import CrossAssetAnalytics


class TestMarketTrapAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("config.json", "r", encoding="utf-8") as f:
            cls.config = json.load(f)
        cls.ohlcv_df, cls.news_df = generate_multi_asset_dataset(n_bars=1200, seed=11)
        cls.sent_engine = FinancialSentimentEngine()
        cls.detector = MarketTrapDetector(cls.config)
        cls.predictor = MarketTrapPredictor(random_state=42)

    def test_sentiment_and_catalyst_routing(self):
        res = self.sent_engine.score_headline(
            "EIA reports surprise 6.5M barrel crude inventory build, crushing demand outlook"
        )
        self.assertEqual(res["routed_asset"], "CRUDE")
        self.assertEqual(res["catalyst_class"], "INVENTORY_SUPPLY")
        self.assertLess(res["sentiment_score"], -0.5)
        self.assertGreater(res["urgency_score"], 0.7)

    def test_liquidity_pools_and_killzone_features(self):
        gold_df = self.ohlcv_df[self.ohlcv_df["asset"] == "GOLD"]
        feats = self.detector.compute_features(gold_df, "GOLD")
        for col in [
            "in_active_killzone",
            "bsl_pool_level",
            "ssl_pool_level",
            "bsl_pool_touches",
            "ssl_pool_touches",
            "sweep_classification",
            "detected_trap_type",
        ]:
            self.assertIn(col, feats.columns)

    def test_end_to_end_training_and_signal_prediction(self):
        enriched_news = self.sent_engine.enrich_news_dataframe(self.news_df)
        import pandas as pd
        parts = [self.detector.compute_features(self.ohlcv_df[self.ohlcv_df["asset"] == a], a) for a in ["CRYPTO", "GOLD", "CRUDE"]]
        fused = self.predictor.fuse_news_and_market(pd.concat(parts, ignore_index=True), enriched_news)
        metrics = self.predictor.train_and_evaluate(fused)
        self.assertGreaterEqual(metrics["roc_auc"], 0.75)

        signal = self.predictor.predict_signal({
            "atr_ratio": 2.9,
            "body_ratio": 0.25,
            "upper_wick_ratio": 0.72,
            "lower_wick_ratio": 0.03,
            "volume_ratio": 2.6,
            "vwap_dist_pct": 0.012,
            "rsi_14": 76.0,
            "in_active_killzone": 1,
            "sin_hour": -0.5,
            "cos_hour": -0.86,
            "bsl_pool_touches": 3,
            "ssl_pool_touches": 1,
            "swept_bsl": 1,
            "swept_ssl": 0,
            "mss_confirmed": 1,
            "sentiment_score": 0.85,
            "urgency_score": 0.90,
            "sentiment_price_divergence": 0.61,
        })
        self.assertIn("trap_probability", signal)
        self.assertIn("recommended_action", signal)


if __name__ == "__main__":
    unittest.main()
