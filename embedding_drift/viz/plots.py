"""Generate 2D scatter plots of embedding snapshots using PCA."""

from datetime import date

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from embedding_drift.core.snapshot import SnapshotStore


def plot_2d_drift(store: SnapshotStore, dates: list[date] | None = None, save_path: str | None = None):
    """Plot 2D PCA projection of embedding snapshots.

    Args:
        store: SnapshotStore containing saved embeddings.
        dates: Specific dates to plot. Defaults to all available.
        save_path: If provided, saves the plot to this path instead of showing it.
    """
    dates = dates or store.list_dates()
    if not dates:
        print("No snapshots found.")
        return

    all_embeddings = []
    for d in dates:
        all_embeddings.append(store.load(d))

    combined = np.vstack(all_embeddings)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    markers = ['o', 'X', 's', 'D', '^', 'v']
    sizes = [120, 80, 80, 80, 80, 80]

    plt.figure(figsize=(10, 7))
    offset = 0
    for i, d in enumerate(dates):
        n = len(all_embeddings[i])
        pts = projected[offset:offset + n]
        plt.scatter(pts[:, 0], pts[:, 1], label=d.isoformat(),
                    alpha=0.7, s=sizes[i % len(sizes)],
                    marker=markers[i % len(markers)], edgecolors='black', linewidths=0.5)
        offset += n

    plt.title("Embedding Space Drift (2D PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
