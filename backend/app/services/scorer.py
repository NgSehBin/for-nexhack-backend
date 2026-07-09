"""
Scorer service — wraps the trained XGBoost model.
Falls back to pure graph-feature scoring if model not yet trained.
"""
import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "fan_in_count", "fan_in_amount", "fan_out_count", "fan_out_amount",
    "pass_through_ratio", "in_degree", "out_degree",
    "proximity_to_mule", "known_mule",
]


class ScorerService:
    def __init__(self):
        self.model = None
        self.features: list[str] = FEATURE_NAMES
        self.explainer = None
        self._load()

    def _load(self):
        model_path = Path(settings.MODEL_PATH)
        features_path = Path(settings.FEATURES_PATH)
        if model_path.exists() and features_path.exists():
            try:
                import joblib, shap
                self.model = joblib.load(model_path)
                self.features = joblib.load(features_path)
                self.explainer = shap.TreeExplainer(self.model.calibrated_classifiers_[0].estimator)
                logger.info("Scorer: XGBoost model loaded")
            except Exception as e:
                logger.warning(f"Scorer: model load failed ({e}) — using graph-only scoring")
        else:
            logger.info("Scorer: no model artifact found — using graph-only scoring")

    def score_node(self, feature_dict: dict) -> dict:
        """Returns {risk_score: 0-100, top_shap: [{feature, value}]}."""
        if self.model is not None:
            return self._score_with_model(feature_dict)
        return self._score_graph_only(feature_dict)

    def _score_with_model(self, feat: dict) -> dict:
        import shap
        X = [[feat.get(f, 0) for f in self.features]]
        prob = float(self.model.predict_proba(X)[0][1])
        score = int(min(100, round(prob * 100)))

        try:
            shap_vals = self.explainer.shap_values(X)[0]
            top = sorted(
                zip(self.features, shap_vals),
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
            top_shap = [{"feature": f, "value": round(float(v), 4)} for f, v in top]
        except Exception:
            top_shap = []

        return {"risk_score": score, "top_shap": top_shap}

    def _score_graph_only(self, feat: dict) -> dict:
        """
        Pure graph formula matching prototype weights:
        fanIn 0.30 | fanOut 0.20 | velocity 0.30 | proximity 0.20
        Returns approximate score without model.
        """
        score = feat.get("score", 0)  # use pre-computed graph score if available
        top_shap = [
            {"feature": "fan_in_count", "value": feat.get("fan_in_count", 0) * 0.30},
            {"feature": "pass_through_ratio", "value": feat.get("pass_through_ratio", 0) * 0.30},
            {"feature": "fan_out_count", "value": feat.get("fan_out_count", 0) * 0.20},
            {"feature": "proximity_to_mule", "value": feat.get("proximity_to_mule", -1) * -0.20},
        ]
        return {"risk_score": score, "top_shap": top_shap}

    def score_cluster(self, node_scores: list[dict]) -> float:
        """
        Cluster score = prototype formula: avg*0.5 + max*0.5 + min(8, count)
        (graph_engine already does this — scorer just re-exposes it for external use)
        """
        scores = [n.get("risk_score", n.get("score", 0)) for n in node_scores]
        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        mx = max(scores)
        return round(min(100, avg * 0.5 + mx * 0.5 + min(8, len(scores))), 1)


# ── Singleton ──────────────────────────────────────────────────────────────
_scorer: ScorerService | None = None


def get_scorer() -> ScorerService:
    global _scorer
    if _scorer is None:
        _scorer = ScorerService()
    return _scorer
