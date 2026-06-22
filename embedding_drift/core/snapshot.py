"""Storage for daily embedding snapshots."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np


class SnapshotStore:
    """Stores and retrieves embedding snapshots indexed by date."""

    def __init__(self, store_dir: str = ".embedding_drift"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, embeddings: np.ndarray, snapshot_date: Optional[date] = None):
        snapshot_date = snapshot_date or date.today()
        day_dir = self.store_dir / snapshot_date.isoformat()
        day_dir.mkdir(exist_ok=True)
        np.save(day_dir / "embeddings.npy", embeddings)
        meta = {"shape": list(embeddings.shape), "saved_at": datetime.now().isoformat()}
        (day_dir / "meta.json").write_text(json.dumps(meta))

    def load(self, snapshot_date: date) -> np.ndarray:
        day_dir = self.store_dir / snapshot_date.isoformat()
        if not day_dir.exists():
            raise FileNotFoundError(f"No snapshot for {snapshot_date.isoformat()}")
        return np.load(day_dir / "embeddings.npy")

    def list_dates(self) -> list[date]:
        dates = []
        for p in self.store_dir.iterdir():
            if p.is_dir():
                try:
                    dates.append(date.fromisoformat(p.name))
                except ValueError:
                    continue
        return sorted(dates)
