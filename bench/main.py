from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import statistics
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


BENCH_RE = re.compile(
    r"BENCH\s+precision=(?P<precision>\w+)\s+iters=(?P<iters>\d+)\s+total_ms=(?P<total>[0-9]*\.?[0-9]+)\s+avg_ms=(?P<avg>[0-9]*\.?[0-9]+)"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_script = repo_root / "scripts" / "run_gpu_test.sh"
    default_out = repo_root / "bench" / "results"

    parser = argparse.ArgumentParser(
        description="Run FP16/BF16 CUDA benchmarks repeatedly and draw statistics plots."
    )
    parser.add_argument("--script", type=Path, default=default_script, help="Benchmark shell script path")
    parser.add_argument("--count", type=int, default=4_194_304, help="Particle/state count per run")
    parser.add_argument("--iters", type=int, default=30, help="Inner iterations for each precision run")
    parser.add_argument("--samples", type=int, default=7, help="Number of repeated benchmark samples")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs before sampling")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Directory to store CSV and plots")
    return parser.parse_args()


def extract_bench(output: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in output.splitlines():
        match = BENCH_RE.search(line)
        if match:
            precision = match.group("precision")
            result[precision] = float(match.group("avg"))
    return result


def run_script(script: Path, count: int, iters: int) -> dict[str, float]:
    command = [str(script), str(count), str(iters)]
    proc = subprocess.run(command, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Benchmark run failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    parsed = extract_bench(proc.stdout)
    if "fp16" not in parsed or "bf16" not in parsed:
        raise RuntimeError(
            "Failed to parse BENCH lines for both precisions.\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return parsed


def summarize(values: list[float]) -> dict[str, float]:
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": stdev,
    }


def save_csv(output_dir: Path, rows: list[dict[str, float]]) -> Path:
    csv_path = output_dir / "bench_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "fp16_avg_ms", "bf16_avg_ms", "bf16_vs_fp16_ratio"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def save_plot(output_dir: Path, fp16: list[float], bf16: list[float]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].boxplot([fp16, bf16], tick_labels=["FP16", "BF16"], patch_artist=True)
    axes[0].set_title("Avg Kernel Time Distribution")
    axes[0].set_ylabel("avg_ms")
    axes[0].grid(axis="y", alpha=0.25)

    x = list(range(1, len(fp16) + 1))
    axes[1].plot(x, fp16, marker="o", label="FP16")
    axes[1].plot(x, bf16, marker="o", label="BF16")
    axes[1].set_title("Per-Sample avg_ms")
    axes[1].set_xlabel("sample")
    axes[1].set_ylabel("avg_ms")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    png_path = output_dir / "bench_plot.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    script = args.script.resolve()
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.warmup):
        run_script(script, args.count, args.iters)

    fp16_values: list[float] = []
    bf16_values: list[float] = []
    rows: list[dict[str, float]] = []

    for i in range(1, args.samples + 1):
        bench = run_script(script, args.count, args.iters)
        fp = bench["fp16"]
        bf = bench["bf16"]
        fp16_values.append(fp)
        bf16_values.append(bf)
        rows.append(
            {
                "sample": i,
                "fp16_avg_ms": fp,
                "bf16_avg_ms": bf,
                "bf16_vs_fp16_ratio": bf / fp,
            }
        )
        print(f"sample {i}/{args.samples}: fp16={fp:.3f} ms, bf16={bf:.3f} ms, ratio={bf / fp:.3f}")

    fp_stats = summarize(fp16_values)
    bf_stats = summarize(bf16_values)
    speed_ratio = fp_stats["mean"] / bf_stats["mean"] if bf_stats["mean"] > 0 else 0.0

    csv_path = save_csv(output_dir, rows)
    png_path = save_plot(output_dir, fp16_values, bf16_values)

    print("\n=== Benchmark Summary ===")
    print(f"count={args.count}, iters={args.iters}, samples={args.samples}, warmup={args.warmup}")
    print(
        "FP16: "
        f"mean={fp_stats['mean']:.3f} ms, median={fp_stats['median']:.3f} ms, "
        f"min={fp_stats['min']:.3f}, max={fp_stats['max']:.3f}, stdev={fp_stats['stdev']:.3f}"
    )
    print(
        "BF16: "
        f"mean={bf_stats['mean']:.3f} ms, median={bf_stats['median']:.3f} ms, "
        f"min={bf_stats['min']:.3f}, max={bf_stats['max']:.3f}, stdev={bf_stats['stdev']:.3f}"
    )
    print(f"Relative speed (FP16 mean / BF16 mean): {speed_ratio:.3f}x")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Plot: {png_path}")


if __name__ == "__main__":
    main()
