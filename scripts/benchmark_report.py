import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-style tables from run metrics CSV files.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_all_metrics(runs_root)
    table2 = _build_latest_table(rows, datasets={"esci"})
    table3 = _build_latest_table(rows, datasets={"beauty", "lastfm"})
    ablation = _build_ablation(rows)

    _write_csv(output_dir / "table2_esci.csv", table2)
    _write_csv(output_dir / "table3_beauty_lastfm.csv", table3)
    _write_csv(output_dir / "dpo_iteration_ablation.csv", ablation)

    print(f"Wrote: {output_dir / 'table2_esci.csv'}")
    print(f"Wrote: {output_dir / 'table3_beauty_lastfm.csv'}")
    print(f"Wrote: {output_dir / 'dpo_iteration_ablation.csv'}")


def _read_all_metrics(runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in runs_root.rglob("metrics.csv"):
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["iteration"] = int(row["iteration"])
                row["value"] = float(row["value"])
                rows.append(row)
    return rows


def _build_latest_table(rows: list[dict], *, datasets: set[str]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row["dataset"] not in datasets:
            continue
        grouped[(row["dataset"], row["metric"])][row["iteration"]] = row["value"]
    output: list[dict] = []
    for (dataset, metric), iteration_values in sorted(grouped.items()):
        latest_iteration = max(iteration_values) if iteration_values else 0
        output.append(
            {
                "dataset": dataset,
                "metric": metric,
                "latest_iteration": latest_iteration,
                "value": iteration_values.get(latest_iteration, 0.0),
            }
        )
    return output


def _build_ablation(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for row in sorted(rows, key=lambda item: (item["dataset"], item["metric"], item["iteration"])):
        output.append(
            {
                "dataset": row["dataset"],
                "metric": row["metric"],
                "iteration": row["iteration"],
                "value": row["value"],
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
