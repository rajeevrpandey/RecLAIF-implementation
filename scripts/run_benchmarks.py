import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command benchmark runner for paper-style tables.")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [
        "configs/experiments/esci.v1.json",
        "configs/experiments/beauty.v1.json",
        "configs/experiments/lastfm.v1.json",
    ]
    for config in configs:
        dataset = Path(config).stem.split(".")[0]
        cmd = [
            sys.executable,
            "scripts/run_iterative_dpo.py",
            "--config",
            config,
            "--dataset-root",
            args.dataset_root,
            "--output-dir",
            str(Path(args.runs_root) / dataset),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd, check=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_report.py",
            "--runs-root",
            args.runs_root,
            "--output-dir",
            args.reports_dir,
        ],
        check=True,
    )
    print(f"Benchmark reports written to: {args.reports_dir}")


if __name__ == "__main__":
    main()
