import pandas as pd


class FinancialSentimentEngine:
    """
    Routes breaking headlines to target assets (Crypto, Gold, Crude), identifies
    catalyst regimes (MACRO, INVENTORY_SUPPLY, FLOW_REGULATORY), and computes
    sentiment scores [-1.0, +1.0] plus market-impact urgency [0.0, 1.0].

    Supports optional drop-in HuggingFace FinBERT ('ProsusAI/finbert') inference.
    """

    ASSET_KEYWORDS = {
        "CRYPTO": ["btc", "bitcoin", "eth", "ethereum", "crypto", "sec", "etf", "exchange", "funding", "whale", "perpetual"],
        "GOLD": ["gold", "xau", "bullion", "fed", "cpi", "inflation", "fomc", "dxy", "yields", "payrolls", "central bank", "treasury"],
        "CRUDE": ["crude", "oil", "wti", "brent", "eia", "opec", "barrel", "refinery", "shipping", "inventory", "api", "distillate"],
    }

    CATALYST_KEYWORDS = {
        "MACRO": ["fed", "cpi", "inflation", "fomc", "payrolls", "rate cut", "rates", "pmi", "yield", "dxy"],
        "INVENTORY_SUPPLY": ["eia", "api", "inventory", "barrel", "opec", "production cut", "shipping", "supply", "distillate"],
        "FLOW_REGULATORY": ["sec", "etf", "outflows", "allocation", "funding", "liquidation", "whale", "exchange"],
    }

    LEXICON = {
        "surging": 0.75, "allocation": 0.80, "buying": 0.70, "cut": 0.65,
        "disruption": 0.75, "shock": 0.70, "lifts": 0.65, "outflows": 0.65,
        "short squeeze": 0.80, "reversal": 0.50,
        "delays": -0.75, "concerns": -0.60, "liquidation": -0.85, "cascade": -0.75,
        "higher for longer": -0.80, "hot cpi": -0.75, "build": -0.75,
        "crushing": -0.80, "contracts": -0.65, "extreme levels": -0.65,
    }

    URGENCY_TERMS = ["unexpectedly", "surprise", "emergency", "cascade", "smash", "sudden", "shock", "spike"]

    def __init__(self, use_finbert: bool = False):
        self.use_finbert = use_finbert
        self._finbert_pipeline = None
        if self.use_finbert:
            from transformers import pipeline
            self._finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def route_asset(self, headline: str) -> str:
        text = headline.lower()
        scores = {
            asset: sum(1 for kw in kws if kw in text)
            for asset, kws in self.ASSET_KEYWORDS.items()
        }
        best_asset = max(scores, key=scores.get)
        return best_asset if scores[best_asset] > 0 else "GENERAL"

    def classify_catalyst(self, headline: str) -> str:
        text = headline.lower()
        scores = {
            cat: sum(1 for kw in kws if kw in text)
            for cat, kws in self.CATALYST_KEYWORDS.items()
        }
        best_cat = max(scores, key=scores.get)
        return best_cat if scores[best_cat] > 0 else "GENERAL"

    def score_headline(self, headline: str) -> dict:
        text = headline.lower()
        asset = self.route_asset(headline)
        catalyst = self.classify_catalyst(headline)

        if self._finbert_pipeline is not None:
            res = self._finbert_pipeline(headline)[0]
            lbl = res["label"].lower()
            conf = float(res["score"])
            sentiment = conf if lbl == "positive" else (-conf if lbl == "negative" else 0.0)
        else:
            matched_scores = [val for term, val in self.LEXICON.items() if term in text]
            sentiment = (
                float(max(-1.0, min(1.0, sum(matched_scores) / len(matched_scores))))
                if matched_scores else 0.0
            )

        urgency_hits = sum(1 for u in self.URGENCY_TERMS if u in text)
        urgency = min(1.0, 0.40 + 0.25 * urgency_hits + 0.30 * abs(sentiment))

        label = "BULLISH" if sentiment > 0.2 else ("BEARISH" if sentiment < -0.2 else "NEUTRAL")
        return {
            "routed_asset": asset,
            "catalyst_class": catalyst,
            "sentiment_score": round(sentiment, 4),
            "sentiment_label": label,
            "urgency_score": round(urgency, 4),
        }

    def enrich_news_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        if news_df.empty:
            return news_df.copy()
        results = news_df["headline"].apply(self.score_headline).apply(pd.Series)
        return pd.concat([news_df.reset_index(drop=True), results], axis=1)
