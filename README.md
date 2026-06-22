# embedding-drift

Detect and visualize embedding space drift in RAG systems and vector databases.

Monitor when your users' queries start drifting away from what your knowledge base can answer — before retrieval quality silently degrades.

## Installation

```bash
pip install embedding-drift
```

## Quick Start

### Python API

```python
import numpy as np
from embedding_drift import DriftDetector, SnapshotStore

# Load your embeddings
baseline = np.load("day1_embeddings.npy")
current = np.load("day2_embeddings.npy")

# Check for drift
detector = DriftDetector()
result = detector.check_drift(baseline, current)

print(result["drifted"])   # True/False
print(result["metrics"])   # centroid_shift, cosine_drift, mmd, spread_change
print(result["alerts"])    # Which thresholds were exceeded
```

### Auto-calibrate thresholds

Don't guess thresholds — calibrate them from your baseline data:

```python
detector = DriftDetector()
detector.calibrate(baseline_embeddings, multiplier=2.0)

# Now check_drift uses calibrated thresholds
result = detector.check_drift(baseline, current)
```

### Save and track snapshots over time

```python
from datetime import date
from embedding_drift import SnapshotStore

store = SnapshotStore(".embedding_drift")

# Save daily snapshots
store.save(todays_embeddings)  # defaults to today's date
store.save(embeddings, snapshot_date=date(2024, 7, 1))

# Load and compare
day1 = store.load(date(2024, 7, 1))
day2 = store.load(date(2024, 7, 2))
```

### Visualize drift

```python
from embedding_drift.viz.plots import plot_2d_drift

store = SnapshotStore(".embedding_drift")
plot_2d_drift(store, save_path="drift.png")
```

## CLI

```bash
# Check drift between two .npy files
embedding-drift check --baseline day1.npy --current day2.npy

# Auto-calibrate and check
embedding-drift check --baseline day1.npy --current day2.npy --calibrate

# Custom thresholds
embedding-drift check --baseline day1.npy --current day2.npy --threshold cosine_drift=0.5 --threshold mmd=0.2

# Generate visualization from snapshot store
embedding-drift plot --store .embedding_drift --output drift.png

# Calibrate thresholds from reference data
embedding-drift calibrate --baseline reference.npy --multiplier 2.5
```

## Metrics

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| `centroid_shift` | Euclidean distance between mean vectors | Close to 0 |
| `cosine_drift` | 1 - cosine similarity of centroids | Close to 0 |
| `mmd` | Maximum Mean Discrepancy (distribution difference) | Close to 0 |
| `spread_change` | Change in average spread from centroid | Close to 0 |

## Use Cases

- **RAG monitoring**: Detect when user queries shift outside your knowledge base
- **Vector DB health**: Track if your embedding index is still relevant
- **Model updates**: Compare embeddings before/after model changes
- **A/B testing**: Measure if different user segments have different query patterns

## Exit Codes (CLI)

- `0` — No drift detected
- `1` — Drift detected (useful for CI/CD pipelines)

## License

MIT
