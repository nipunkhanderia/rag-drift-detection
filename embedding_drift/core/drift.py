"""Drift detection metrics for embedding spaces."""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


DEFAULT_THRESHOLDS = {
    "centroid_shift": 0.5,
    "cosine_drift": 0.02,
    "mmd": 0.05,
    "spread_change": 0.3,
}


class DriftDetector:
    """Computes drift metrics between reference and current embeddings."""

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()

    @staticmethod
    def centroid_shift(ref: np.ndarray, cur: np.ndarray) -> float:
        return float(np.linalg.norm(ref.mean(axis=0) - cur.mean(axis=0)))

    @staticmethod
    def cosine_drift(ref: np.ndarray, cur: np.ndarray) -> float:
        ref_mean, cur_mean = ref.mean(axis=0), cur.mean(axis=0)
        cos_sim = np.dot(ref_mean, cur_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(cur_mean) + 1e-10)
        return float(1 - cos_sim)

    @staticmethod
    def mmd(ref: np.ndarray, cur: np.ndarray, gamma: float = 1.0) -> float:
        xx = rbf_kernel(ref, ref, gamma=gamma).mean()
        yy = rbf_kernel(cur, cur, gamma=gamma).mean()
        xy = rbf_kernel(ref, cur, gamma=gamma).mean()
        return float(xx + yy - 2 * xy)

    @staticmethod
    def spread_change(ref: np.ndarray, cur: np.ndarray) -> float:
        ref_spread = np.linalg.norm(ref - ref.mean(axis=0), axis=1).mean()
        cur_spread = np.linalg.norm(cur - cur.mean(axis=0), axis=1).mean()
        return float(cur_spread - ref_spread)

    def compute_all(self, ref: np.ndarray, cur: np.ndarray) -> dict:
        return {
            "centroid_shift": self.centroid_shift(ref, cur),
            "cosine_drift": self.cosine_drift(ref, cur),
            "mmd": self.mmd(ref, cur),
            "spread_change": self.spread_change(ref, cur),
        }

    def check_drift(self, ref: np.ndarray, cur: np.ndarray) -> dict:
        """Returns metrics + whether drift was detected."""
        metrics = self.compute_all(ref, cur)
        alerts = []
        for metric, value in metrics.items():
            threshold = self.thresholds.get(metric, float("inf"))
            if abs(value) >= threshold:
                alerts.append({"metric": metric, "value": value, "threshold": threshold})
        return {"metrics": metrics, "drifted": len(alerts) > 0, "alerts": alerts}
