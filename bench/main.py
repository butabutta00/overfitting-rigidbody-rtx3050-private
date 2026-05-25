from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import statistics
import subprocess
import time
from pathlib import Path

KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run CUDA and C# benchmarks with the same 1D spring-mass-damper I/O spec."
    )

    parser.add_argument("--samples", type=int, default=10, help="Number of measured runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs before sampling")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "bench" / "results", help="Output directory")

    parser.add_argument("--skip-cuda-build", action="store_true", help="Skip CUDA build script")
    parser.add_argument(
        "--cuda-build-script",
        type=Path,
        default=repo_root / "scripts" / "build-cuda-standalone-test.sh",
        help="CUDA build script path",
    )
    parser.add_argument("--cuda-arch", type=str, default="86-real;86-virtual", help="CUDA arch arg")
    parser.add_argument("--cuda-binary", type=Path, default=repo_root / "cuda" / "build" / "main", help="CUDA binary path")

    parser.add_argument("--skip-csharp-build", action="store_true", help="Skip C# build")
    parser.add_argument(
        "--csharp-project",
        type=Path,
        default=repo_root / "cpu_csharp" / "CpuMassSpringDamper" / "CpuMassSpringDamper.csproj",
        help="C# project path",
    )

    parser.add_argument("--position", type=float, default=1.0, help="Initial position")
    parser.add_argument("--velocity", type=float, default=0.0, help="Initial velocity")
    parser.add_argument("--dt", type=float, default=0.016, help="Time-step")
    parser.add_argument("--mass", type=float, default=1.0, help="Mass")
    parser.add_argument("--stiffness", type=float, default=120.0, help="Spring stiffness")
    parser.add_argument("--damping", type=float, default=0.2, help="Spring damping")
    parser.add_argument("--steps", type=int, default=200, help="Integration steps")
    return parser.parse_args()


def run_command(command: list[str], fail_prefix: str) -> tuple[float, str]:
    start = time.perf_counter()
    proc = subprocess.run(command, check=False, text=True, capture_output=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(
            f"{fail_prefix}\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return elapsed_ms, proc.stdout


def summarize(values: list[float]) -> dict[str, float]:
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": stdev,
    }


def parse_kv_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in KEY_VALUE_RE.finditer(stdout):
        metrics[match.group("key")] = float(match.group("value"))
    return metrics


def require_metrics(label: str, metrics: dict[str, float], required: list[str], stdout: str) -> dict[str, float]:
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f"Failed to parse {label} metrics: missing {missing}\nSTDOUT:\n{stdout}")
    return metrics


def resolve_cuda_binary(binary: Path) -> Path:
    resolved = binary.resolve()
    if resolved.exists():
        return resolved

    fallback = resolved.parent / "Release" / resolved.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"CUDA binary not found: {resolved}")


def build_cuda(args: argparse.Namespace) -> None:
    if args.skip_cuda_build:
        return
    build_script = args.cuda_build_script.resolve()
    if not build_script.exists():
        raise FileNotFoundError(f"CUDA build script not found: {build_script}")
    run_command(["bash", str(build_script), args.cuda_arch], "CUDA build failed.")


def build_csharp(args: argparse.Namespace) -> None:
    if args.skip_csharp_build:
        return
    project = args.csharp_project.resolve()
    if not project.exists():
        raise FileNotFoundError(f"C# project not found: {project}")
    run_command(["dotnet", "build", "-c", "Release", str(project)], "C# build failed.")


def benchmark_args(args: argparse.Namespace) -> list[str]:
    return [
        "--position",
        str(args.position),
        "--velocity",
        str(args.velocity),
        "--dt",
        str(args.dt),
        "--mass",
        str(args.mass),
        "--stiffness",
        str(args.stiffness),
        "--damping",
        str(args.damping),
        "--steps",
        str(args.steps),
    ]


def run_cuda(args: argparse.Namespace, cuda_binary: Path) -> tuple[float, dict[str, float]]:
    command = [str(cuda_binary), *benchmark_args(args)]
    wall_ms, stdout = run_command(command, "CUDA run failed.")
    metrics = parse_kv_metrics(stdout)
    require_metrics("CUDA", metrics, ["elapsed_ms", "output_x", "output_v", "checksum"], stdout)
    return wall_ms, metrics


def run_csharp(args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    project = args.csharp_project.resolve()
    command = [
        "dotnet",
        "run",
        "-c",
        "Release",
        "--project",
        str(project),
        "--",
        *benchmark_args(args),
    ]
    wall_ms, stdout = run_command(command, "C# run failed.")
    metrics = parse_kv_metrics(stdout)
    require_metrics("C#", metrics, ["elapsed_ms", "output_x", "output_v", "checksum"], stdout)
    return wall_ms, metrics


def save_csv(output_dir: Path, rows: list[dict[str, float]]) -> Path:
    csv_path = output_dir / "cuda_vs_csharp_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "cuda_wall_ms",
                "cuda_elapsed_ms",
                "cuda_output_x",
                "cuda_output_v",
                "cuda_checksum",
                "csharp_wall_ms",
                "csharp_elapsed_ms",
                "csharp_output_x",
                "csharp_output_v",
                "csharp_checksum",
                "wall_speedup_cuda_over_csharp",
                "elapsed_speedup_cuda_over_csharp",
                "abs_diff_output_x",
                "abs_diff_output_v",
                "abs_diff_checksum",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.warmup < 0:
        raise ValueError("samples must be >= 1 and warmup must be >= 0")
    if args.steps < 1:
        raise ValueError("steps must be >= 1")

    build_cuda(args)
    build_csharp(args)
    cuda_binary = resolve_cuda_binary(args.cuda_binary)

    for _ in range(args.warmup):
        run_cuda(args, cuda_binary)
        run_csharp(args)

    rows: list[dict[str, float]] = []
    cuda_elapsed_values: list[float] = []
    csharp_elapsed_values: list[float] = []
    cuda_wall_values: list[float] = []
    csharp_wall_values: list[float] = []
    diff_checksum_values: list[float] = []

    for i in range(1, args.samples + 1):
        cuda_wall_ms, cuda_metrics = run_cuda(args, cuda_binary)
        csharp_wall_ms, csharp_metrics = run_csharp(args)

        cuda_elapsed_ms = cuda_metrics["elapsed_ms"]
        csharp_elapsed_ms = csharp_metrics["elapsed_ms"]
        cuda_x = cuda_metrics["output_x"]
        cuda_v = cuda_metrics["output_v"]
        cuda_checksum = cuda_metrics["checksum"]
        csharp_x = csharp_metrics["output_x"]
        csharp_v = csharp_metrics["output_v"]
        csharp_checksum = csharp_metrics["checksum"]

        wall_ratio = cuda_wall_ms / csharp_wall_ms if csharp_wall_ms > 0.0 else 0.0
        elapsed_ratio = cuda_elapsed_ms / csharp_elapsed_ms if csharp_elapsed_ms > 0.0 else 0.0
        diff_x = abs(cuda_x - csharp_x)
        diff_v = abs(cuda_v - csharp_v)
        diff_checksum = abs(cuda_checksum - csharp_checksum)

        rows.append(
            {
                "sample": i,
                "cuda_wall_ms": cuda_wall_ms,
                "cuda_elapsed_ms": cuda_elapsed_ms,
                "cuda_output_x": cuda_x,
                "cuda_output_v": cuda_v,
                "cuda_checksum": cuda_checksum,
                "csharp_wall_ms": csharp_wall_ms,
                "csharp_elapsed_ms": csharp_elapsed_ms,
                "csharp_output_x": csharp_x,
                "csharp_output_v": csharp_v,
                "csharp_checksum": csharp_checksum,
                "wall_speedup_cuda_over_csharp": wall_ratio,
                "elapsed_speedup_cuda_over_csharp": elapsed_ratio,
                "abs_diff_output_x": diff_x,
                "abs_diff_output_v": diff_v,
                "abs_diff_checksum": diff_checksum,
            }
        )

        cuda_wall_values.append(cuda_wall_ms)
        csharp_wall_values.append(csharp_wall_ms)
        cuda_elapsed_values.append(cuda_elapsed_ms)
        csharp_elapsed_values.append(csharp_elapsed_ms)
        diff_checksum_values.append(diff_checksum)

        print(
            f"sample {i}/{args.samples}: "
            f"cuda_elapsed={cuda_elapsed_ms:.3f}ms, "
            f"csharp_elapsed={csharp_elapsed_ms:.3f}ms, "
            f"abs_diff_checksum={diff_checksum:.6f}"
        )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_csv(output_dir, rows)

    cuda_elapsed_stats = summarize(cuda_elapsed_values)
    csharp_elapsed_stats = summarize(csharp_elapsed_values)
    cuda_wall_stats = summarize(cuda_wall_values)
    csharp_wall_stats = summarize(csharp_wall_values)
    diff_checksum_stats = summarize(diff_checksum_values)

    print("\n=== CUDA vs C# Common-Spec Benchmark Summary ===")
    print(
        f"samples={args.samples}, warmup={args.warmup}, "
        f"x0={args.position}, v0={args.velocity}, dt={args.dt}, "
        f"mass={args.mass}, stiffness={args.stiffness}, damping={args.damping}, steps={args.steps}"
    )
    print(
        "cuda elapsed(ms): "
        f"mean={cuda_elapsed_stats['mean']:.3f}, median={cuda_elapsed_stats['median']:.3f}, "
        f"min={cuda_elapsed_stats['min']:.3f}, max={cuda_elapsed_stats['max']:.3f}, stdev={cuda_elapsed_stats['stdev']:.3f}"
    )
    print(
        "csharp elapsed(ms): "
        f"mean={csharp_elapsed_stats['mean']:.3f}, median={csharp_elapsed_stats['median']:.3f}, "
        f"min={csharp_elapsed_stats['min']:.3f}, max={csharp_elapsed_stats['max']:.3f}, stdev={csharp_elapsed_stats['stdev']:.3f}"
    )
    print(
        "cuda wall(ms): "
        f"mean={cuda_wall_stats['mean']:.3f}, median={cuda_wall_stats['median']:.3f}, "
        f"min={cuda_wall_stats['min']:.3f}, max={cuda_wall_stats['max']:.3f}, stdev={cuda_wall_stats['stdev']:.3f}"
    )
    print(
        "csharp wall(ms): "
        f"mean={csharp_wall_stats['mean']:.3f}, median={csharp_wall_stats['median']:.3f}, "
        f"min={csharp_wall_stats['min']:.3f}, max={csharp_wall_stats['max']:.3f}, stdev={csharp_wall_stats['stdev']:.3f}"
    )
    print(
        "abs diff checksum: "
        f"mean={diff_checksum_stats['mean']:.6f}, median={diff_checksum_stats['median']:.6f}, "
        f"min={diff_checksum_stats['min']:.6f}, max={diff_checksum_stats['max']:.6f}, stdev={diff_checksum_stats['stdev']:.6f}"
    )
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
