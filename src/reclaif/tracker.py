from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrackRow:
    run_id: str
    dataset: str
    split: str
    stage: str
    iteration: int
    metric: str
    value: float


class CSVExperimentTracker:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists():
            with self.output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["run_id", "dataset", "split", "stage", "iteration", "metric", "value"])

    def log(self, row: TrackRow) -> None:
        with self.output_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    row.run_id,
                    row.dataset,
                    row.split,
                    row.stage,
                    row.iteration,
                    row.metric,
                    f"{row.value:.6f}",
                ]
            )
