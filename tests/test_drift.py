"""Tests for embedding_drift package."""

import tempfile
from datetime import date

import numpy as np
import pytest

from embedding_drift import DriftDetector, SnapshotStore


class TestDriftDetector:
    def test_no_drift_identical(self):
        emb = np.random.randn(20, 128).astype(np.float32)
        detector = DriftDetector()
        result = detector.check_drift(emb, emb)
        assert not result["drifted"]
        assert result["metrics"]["centroid_shift"] == 0.0
        assert result["metrics"]["mmd"] == 0.0

    def test_drift_detected_shifted(self):
        ref = np.random.randn(20, 128).astype(np.float32)
        cur = ref + 5.0  # large shift
        detector = DriftDetector()
        result = detector.check_drift(ref, cur)
        assert result["drifted"]
        assert result["metrics"]["centroid_shift"] > 0.5

    def test_custom_thresholds(self):
        ref = np.random.randn(20, 128).astype(np.float32)
        cur = ref + 0.01
        # Very loose thresholds — should not trigger
        detector = DriftDetector(thresholds={"centroid_shift": 100, "cosine_drift": 100, "mmd": 100, "spread_change": 100})
        result = detector.check_drift(ref, cur)
        assert not result["drifted"]

    def test_calibrate(self):
        ref = np.random.randn(50, 64).astype(np.float32)
        detector = DriftDetector()
        thresholds = detector.calibrate(ref, n_splits=5, multiplier=2.0)
        assert all(v > 0 for v in thresholds.values())

    def test_compute_all_returns_all_metrics(self):
        ref = np.random.randn(10, 32).astype(np.float32)
        cur = np.random.randn(10, 32).astype(np.float32)
        detector = DriftDetector()
        metrics = detector.compute_all(ref, cur)
        assert set(metrics.keys()) == {"centroid_shift", "cosine_drift", "mmd", "spread_change"}


class TestSnapshotStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            emb = np.random.randn(10, 64).astype(np.float32)
            d = date(2024, 1, 15)
            store.save(emb, snapshot_date=d)
            loaded = store.load(d)
            np.testing.assert_array_equal(emb, loaded)

    def test_list_dates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            dates = [date(2024, 1, 1), date(2024, 1, 3), date(2024, 1, 2)]
            for d in dates:
                store.save(np.random.randn(5, 32).astype(np.float32), snapshot_date=d)
            listed = store.list_dates()
            assert listed == sorted(dates)

    def test_load_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            with pytest.raises(FileNotFoundError):
                store.load(date(2099, 12, 31))
