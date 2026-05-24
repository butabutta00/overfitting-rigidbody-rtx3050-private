from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import statistics
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_RE = re.compile(r"Output\s*->\s*x=(?P<x>-?[0-9]*\.?[0-9]+)\s*,\s*v=(?P<v>-?[0-9]*\.?[0-9]+)")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_build_script = repo_root / "scripts" / "build-cuda-standalone-test.sh"
    default_binary = repo_root / "cuda" / "build" / "main"
    default_out = repo_root / "bench" / "results"

    parser = argparse.ArgumentParser(
        description="Run CUDA main.cu entrypoint repeatedly and collect runtime/state statistics."
    )
    parser.add_argument("--build-script", type=Path, default=default_build_script, help="Build script path")
    parser.add_argument("--binary", type=Path, default=default_binary, help="Built CUDA binary path")
    parser.add_argument("--cuda-arch", type=str, default="86-real;86-virtual", help="CUDA architectures for build script")

    parser.add_argument("--position", type=float, default=1.0, help="Initial position")
    parser.add_argument("--velocity", type=float, default=0.0, help="Initial velocity")
    parser.add_argument("--dt", type=float, default=0.016, help="Time step")
    parser.add_argument("--mass", type=float, default=1.0, help="Mass")
    parser.add_argument("--stiffness", type=float, default=120.0, help="Spring stiffness")
    parser.add_argument("--damping", type=float, default=0.2, help="Damping")
    parser.add_argument("--steps", type=int, default=8, help="Implicit solver steps")

    parser.add_argument("--samples", type=int, default=10, help="Number of measured runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs before sampling")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Directory to store CSV and plots")
    return parser.parse_args()


def build_binary(build_script: Path, cuda_arch: str) -> None:
    command = ["bash", str(build_script), cuda_arch]
    proc = subprocess.run(command, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Build failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )


def parse_output(stdout: str) -> tuple[float, float]:
    for line in stdout.splitlines():
        match = OUTPUT_RE.search(line)
        if match:
            return float(match.group("x")), float(match.group("v"))
    raise RuntimeError(f"Failed to parse output line from CUDA binary.\nSTDOUT:\n{stdout}")


def run_binary(
    binary: Path,
    position: float,
    velocity: float,
    dt_value: float,
    mass: float,
    stiffness: float,
    damping: float,
    steps: int,
) -> tuple[float, float, float]:
    command = [
        str(binary),
        str(position),
        str(velocity),
        str(dt_value),
        str(mass),
        str(stiffness),
        str(damping),
        str(steps),
    ]

    start = time.perf_counter()
    proc = subprocess.run(command, check=False, text=True, capture_output=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if proc.returncode != 0:
        raise RuntimeError(
            "CUDA run failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    out_x, out_v = parse_output(proc.stdout)
    return elapsed_ms, out_x, out_v


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
    csv_path = output_dir / "main_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample", "elapsed_ms", "output_x", "output_v"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def save_plot(output_dir: Path, elapsed_values: list[float], x_values: list[float], v_values: list[float]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    sample_index = list(range(1, len(elapsed_values) + 1))

    axes[0].plot(sample_index, elapsed_values, marker="o", label="elapsed_ms")
    axes[0].set_title("CUDA main Runtime")
    axes[0].set_xlabel("sample")
    axes[0].set_ylabel("ms")
    axes[0].grid(alpha=0.25)

    axes[1].plot(sample_index, x_values, marker="o", label="output_x")
    axes[1].plot(sample_index, v_values, marker="o", label="output_v")
    axes[1].set_title("Output State")
    axes[1].set_xlabel("sample")
    axes[1].set_ylabel("value")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    png_path = output_dir / "main_plot.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()

    build_script = args.build_script.resolve()
    if not build_script.exists():
        raise FileNotFoundError(f"Build script not found: {build_script}")

    build_binary(build_script, args.cuda_arch)

    binary = args.binary.resolve()
    if not binary.exists():
        fallback = binary.parent / "Release" / binary.name
        if fallback.exists():
            binary = fallback
        else:
            raise FileNotFoundError(f"CUDA binary not found: {binary}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.warmup):
        run_binary(
            binary,
            args.position,
            args.velocity,
            args.dt,
            args.mass,
            args.stiffness,
            args.damping,
            args.steps,
        )

    elapsed_values: list[float] = []
    x_values: list[float] = []
    v_values: list[float] = []
    rows: list[dict[str, float]] = []

    for i in range(1, args.samples + 1):
        elapsed_ms, out_x, out_v = run_binary(
            binary,
            args.position,
            args.velocity,
            args.dt,
            args.mass,
            args.stiffness,
            args.damping,
            args.steps,
        )
        elapsed_values.append(elapsed_ms)
        x_values.append(out_x)
        v_values.append(out_v)

        rows.append(
            {
                "sample": i,
                "elapsed_ms": elapsed_ms,
                "output_x": out_x,
                "output_v": out_v,
            }
        )
        print(f"sample {i}/{args.samples}: elapsed={elapsed_ms:.3f} ms, x={out_x:.6f}, v={out_v:.6f}")

    elapsed_stats = summarize(elapsed_values)
    x_stats = summarize(x_values)
    v_stats = summarize(v_values)

    csv_path = save_csv(output_dir, rows)
    png_path = save_plot(output_dir, elapsed_values, x_values, v_values)

    print("\n=== main.cu Benchmark Summary ===")
    print(
        f"samples={args.samples}, warmup={args.warmup}, "
        f"x0={args.position}, v0={args.velocity}, dt={args.dt}, m={args.mass}, "
        f"k={args.stiffness}, c={args.damping}, steps={args.steps}"
    )
    print(
        "runtime(ms): "
        f"mean={elapsed_stats['mean']:.3f}, median={elapsed_stats['median']:.3f}, "
        f"min={elapsed_stats['min']:.3f}, max={elapsed_stats['max']:.3f}, stdev={elapsed_stats['stdev']:.3f}"
    )
    print(
        "output_x: "
        f"mean={x_stats['mean']:.6f}, median={x_stats['median']:.6f}, "
        f"min={x_stats['min']:.6f}, max={x_stats['max']:.6f}, stdev={x_stats['stdev']:.6f}"
    )
    print(
        "output_v: "
        f"mean={v_stats['mean']:.6f}, median={v_stats['median']:.6f}, "
        f"min={v_stats['min']:.6f}, max={v_stats['max']:.6f}, stdev={v_stats['stdev']:.6f}"
    )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Plot: {png_path}")


if __name__ == "__main__":
    main()
