import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


class MarketTrapPredictor:
    """
    Fuses breaking news sentiment with institutional liquidity pool, killzone,
    and candle microstructure features using a soft-voting ensemble classifier
    to predict trap probability and actionable trade signals.
    """

    FEATURE_COLS = [
        "atr_ratio",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "volume_ratio",
        "vwap_dist_pct",
        "rsi_14",
        "in_active_killzone",
        "sin_hour",
        "cos_hour",
        "bsl_pool_touches",
        "ssl_pool_touches",
        "swept_bsl",
        "swept_ssl",
        "sentiment_score",
        "urgency_score",
        "sentiment_price_divergence",
    ]

    def __init__(self, random_state: int = 42):
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        )
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=4,
            random_state=random_state,
        )
        self.model = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
            weights=[0.5, 0.5],
        )
        self.is_trained = False
        self._rf_ref = rf

    @staticmethod
    def fuse_news_and_market(market_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        if news_df.empty:
            merged = market_df.copy()
            merged["headline"] = ""
            merged["sentiment_score"] = 0.0
            merged["urgency_score"] = 0.0
            merged["catalyst_class"] = "NONE"
        else:
            merged = pd.merge(
                market_df,
                news_df[["timestamp", "asset", "headline", "sentiment_score", "urgency_score", "catalyst_class"]],
                on=["timestamp", "asset"],
                how="left",
            )
            merged["headline"] = merged["headline"].fillna("")
            merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
            merged["urgency_score"] = merged["urgency_score"].fillna(0.0)
            merged["catalyst_class"] = merged["catalyst_class"].fillna("NONE")

        # Sentiment-Price Divergence: strong one-sided news + opposing wick rejection
        price_dir = np.sign(merged["close"] - merged["open"])
        wick_rejection = np.where(
            price_dir >= 0, merged["upper_wick_ratio"], merged["lower_wick_ratio"]
        )
        merged["sentiment_price_divergence"] = merged["sentiment_score"].abs() * wick_rejection
        return merged

    def train_and_evaluate(self, fused_df: pd.DataFrame, test_size: float = 0.25) -> dict:
        X = fused_df[self.FEATURE_COLS]
        y = fused_df["event_label"]

        stratify_arg = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_arg, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        roc_auc = roc_auc_score(y_test, y_prob)

        fitted_rf = self.model.named_estimators_["rf"]
        importances = dict(
            sorted(
                zip(self.FEATURE_COLS, fitted_rf.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        return {
            "test_samples": int(len(y_test)),
            "positive_traps_in_test": int(y_test.sum()),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "top_features": {k: round(float(v), 4) for k, v in list(importances.items())[:7]},
        }

    def predict_signal(self, feature_row: dict) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict_signal.")

        row_complete = {col: feature_row.get(col, 0.0) for col in self.FEATURE_COLS}
        df_row = pd.DataFrame([row_complete])[self.FEATURE_COLS]
        prob = float(self.model.predict_proba(df_row)[0, 1])

        mss_confirmed = bool(feature_row.get("mss_confirmed", 1))

        if prob >= 0.60:
            if not mss_confirmed:
                action = "WAIT_FOR_MSS_CONFIRMATION"
                regime = "PENDING_LIQUIDITY_SWEEP"
            elif feature_row.get("swept_bsl", 0) == 1 or feature_row.get("sentiment_score", 0) > 0:
                action = "FADE_BSL_BULL_TRAP_SHORT"
                regime = "BULL_TRAP_REVERSAL"
            else:
                action = "FADE_SSL_BEAR_TRAP_LONG"
                regime = "BEAR_TRAP_REVERSAL"
        elif feature_row.get("urgency_score", 0) > 0.60 and prob < 0.35:
            action = "FOLLOW_BREAKOUT_MOMENTUM"
            regime = "VALID_BREAKOUT_TREND"
        else:
            action = "HOLD_NO_EDGE"
            regime = "NO_EDGE_CHOP"

        confidence = "HIGH" if (prob >= 0.78 or prob <= 0.15) else ("MEDIUM" if prob >= 0.60 else "LOW")
        return {
            "trap_probability": round(prob, 4),
            "predicted_regime": regime,
            "recommended_action": action,
            "confidence_tier": confidence,
        }
