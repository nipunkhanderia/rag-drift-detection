"""Embedding drift detection and visualization."""

from embedding_drift.core.snapshot import SnapshotStore
from embedding_drift.core.drift import DriftDetector

__version__ = "0.1.0"
__all__ = ["SnapshotStore", "DriftDetector"]
