import json
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

from src.sentiment_engine import FinancialSentimentEngine
from src.data_generator import generate_multi_asset_dataset
from src.trap_detector import MarketTrapDetector
from src.predictor import MarketTrapPredictor
from src.analytics import CrossAssetAnalytics

app = FastAPI(
    title="Market Trap AI API",
    description="Institutional Liquidity Sweep, Breaking News Sentiment & Trap Prediction Engine",
    version="2.0.0",
)

with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

sentiment_engine = FinancialSentimentEngine()
detector = MarketTrapDetector(CONFIG)
predictor = MarketTrapPredictor()

# Bootstrap model & analytics on startup
ohlcv_df, news_df = generate_multi_asset_dataset(n_bars=1500, seed=42)
enriched_news = sentiment_engine.enrich_news_dataframe(news_df)
featured_parts = [
    detector.compute_features(ohlcv_df[ohlcv_df["asset"] == a], a)
    for a in ["CRYPTO", "GOLD", "CRUDE"]
]
fused_data = predictor.fuse_news_and_market(pd.concat(featured_parts, ignore_index=True), enriched_news)
MODEL_METRICS = predictor.train_and_evaluate(fused_data)
ANALYTICS_REPORT = CrossAssetAnalytics.evaluate_trap_strategy(fused_data)


class HeadlineRequest(BaseModel):
    headline: str


class LiveSignalRequest(BaseModel):
    asset: str
    timestamp_utc: str = "2026-11-19T13:35:00"
    headline: str
    atr_ratio: float
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    volume_ratio: float
    vwap_dist_pct: float = 0.004
    rsi_14: float = 68.0
    bsl_pool_touches: int = 2
    ssl_pool_touches: int = 1
    swept_bsl: int = 1
    swept_ssl: int = 0
    mss_confirmed: int = 1


@app.get("/health")
def health_check():
    return {"status": "online", "version": "2.0.0", "model_trained": predictor.is_trained}


@app.post("/analyze/sentiment")
def analyze_sentiment(req: HeadlineRequest):
    return sentiment_engine.score_headline(req.headline)


@app.post("/predict/trap")
def predict_trap(req: LiveSignalRequest):
    asset = req.asset.upper()
    sent = sentiment_engine.score_headline(req.headline)
    ts = datetime.fromisoformat(req.timestamp_utc)
    hour_float = ts.hour + ts.minute / 60.0

    killzones = CONFIG["assets"].get(asset, {}).get("killzones_utc", [])
    in_kz = 0
    for kz in killzones:
        s, e = kz["start_hour"], kz["end_hour"]
        if (s <= e and s <= ts.hour < e) or (s > e and (ts.hour >= s or ts.hour < e)):
            in_kz = 1
            break

    wick_rej = req.upper_wick_ratio if req.swept_bsl else req.lower_wick_ratio
    features = {
        "atr_ratio": req.atr_ratio,
        "body_ratio": req.body_ratio,
        "upper_wick_ratio": req.upper_wick_ratio,
        "lower_wick_ratio": req.lower_wick_ratio,
        "volume_ratio": req.volume_ratio,
        "vwap_dist_pct": req.vwap_dist_pct,
        "rsi_14": req.rsi_14,
        "in_active_killzone": in_kz,
        "sin_hour": float(np.sin(2 * np.pi * hour_float / 24.0)),
        "cos_hour": float(np.cos(2 * np.pi * hour_float / 24.0)),
        "bsl_pool_touches": req.bsl_pool_touches,
        "ssl_pool_touches": req.ssl_pool_touches,
        "swept_bsl": req.swept_bsl,
        "swept_ssl": req.swept_ssl,
        "mss_confirmed": req.mss_confirmed,
        "sentiment_score": sent["sentiment_score"],
        "urgency_score": sent["urgency_score"],
        "sentiment_price_divergence": abs(sent["sentiment_score"]) * wick_rej,
    }
    prediction = predictor.predict_signal(features)
    return {
        "asset": asset,
        "timestamp_utc": req.timestamp_utc,
        "in_active_killzone": bool(in_kz),
        "sentiment": sent,
        "prediction": prediction,
    }


@app.get("/analytics/evaluation")
def get_evaluation():
    return {"ml_evaluation": MODEL_METRICS, "trading_analytics": ANALYTICS_REPORT}


@app.get("/analytics/killzones")
def get_killzone_config():
    return {asset: cfg["killzones_utc"] for asset, cfg in CONFIG["assets"].items()}
