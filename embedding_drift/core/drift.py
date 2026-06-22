"""Drift detection metrics for embedding spaces."""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


DEFAULT_THRESHOLDS = {
    "centroid_shift": 0.5,
    "cosine_drift": 0.5,
    "mmd": 0.2,
    "spread_change": 0.3,
}


class DriftDetector:
    """Detect embedding drift between a reference and current distribution.

    Args:
        thresholds: Dict of metric_name -> threshold value. If a metric exceeds
            its threshold, drift is flagged. Use calibrate() to auto-set thresholds.

    Example:
        >>> detector = DriftDetector()
        >>> result = detector.check_drift(baseline_embeddings, current_embeddings)
        >>> if result["drifted"]:
        ...     print("Drift detected!", result["alerts"])
    """

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()

    @staticmethod
    def centroid_shift(ref: np.ndarray, cur: np.ndarray) -> float:
        """Euclidean distance between centroids of two distributions."""
        return float(np.linalg.norm(ref.mean(axis=0) - cur.mean(axis=0)))

    @staticmethod
    def cosine_drift(ref: np.ndarray, cur: np.ndarray) -> float:
        """1 - cosine similarity between mean vectors. 0 = same direction, 1 = orthogonal."""
        ref_mean, cur_mean = ref.mean(axis=0), cur.mean(axis=0)
        cos_sim = np.dot(ref_mean, cur_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(cur_mean) + 1e-10)
        return float(1 - cos_sim)

    @staticmethod
    def mmd(ref: np.ndarray, cur: np.ndarray, gamma: float = 1.0) -> float:
        """Maximum Mean Discrepancy with RBF kernel. 0 = identical distributions."""
        xx = rbf_kernel(ref, ref, gamma=gamma).mean()
        yy = rbf_kernel(cur, cur, gamma=gamma).mean()
        xy = rbf_kernel(ref, cur, gamma=gamma).mean()
        return float(xx + yy - 2 * xy)

    @staticmethod
    def spread_change(ref: np.ndarray, cur: np.ndarray) -> float:
        """Change in average distance from centroid. Positive = more spread."""
        ref_spread = np.linalg.norm(ref - ref.mean(axis=0), axis=1).mean()
        cur_spread = np.linalg.norm(cur - cur.mean(axis=0), axis=1).mean()
        return float(cur_spread - ref_spread)

    def compute_all(self, ref: np.ndarray, cur: np.ndarray) -> dict:
        """Compute all drift metrics without applying thresholds."""
        return {
            "centroid_shift": self.centroid_shift(ref, cur),
            "cosine_drift": self.cosine_drift(ref, cur),
            "mmd": self.mmd(ref, cur),
            "spread_change": self.spread_change(ref, cur),
        }

    def check_drift(self, ref: np.ndarray, cur: np.ndarray) -> dict:
        """Check if drift occurred between reference and current embeddings.

        Returns:
            Dict with keys: metrics, drifted (bool), alerts (list of triggered thresholds)
        """
        metrics = self.compute_all(ref, cur)
        alerts = []
        for metric, value in metrics.items():
            threshold = self.thresholds.get(metric, float("inf"))
            if abs(value) >= threshold:
                alerts.append({"metric": metric, "value": value, "threshold": threshold})
        return {"metrics": metrics, "drifted": len(alerts) > 0, "alerts": alerts}

    def calibrate(self, reference_embeddings: np.ndarray, n_splits: int = 5, multiplier: float = 2.0):
        """Auto-calibrate thresholds from reference data by splitting it and measuring internal variance.

        Splits the reference embeddings into n_splits folds, computes drift metrics between
        each pair, and sets thresholds at multiplier * max observed value.

        Args:
            reference_embeddings: Known-good baseline embeddings.
            n_splits: Number of folds to split reference data into.
            multiplier: How many times above max internal variance to set threshold.
        """
        n = len(reference_embeddings)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // n_splits

        max_metrics = {k: 0.0 for k in DEFAULT_THRESHOLDS}

        for i in range(n_splits):
            for j in range(i + 1, n_splits):
                fold_a = reference_embeddings[indices[i * fold_size:(i + 1) * fold_size]]
                fold_b = reference_embeddings[indices[j * fold_size:(j + 1) * fold_size]]
                if len(fold_a) < 2 or len(fold_b) < 2:
                    continue
                metrics = self.compute_all(fold_a, fold_b)
                for k, v in metrics.items():
                    max_metrics[k] = max(max_metrics[k], abs(v))

        self.thresholds = {k: v * multiplier for k, v in max_metrics.items()}
        return self.thresholds
