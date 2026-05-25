from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import statistics
import subprocess
import time
from pathlib import Path

CUDA_OUTPUT_RE = re.compile(r"Output\s*->\s*x=(?P<x>-?[0-9]*\.?[0-9]+)\s*,\s*v=(?P<v>-?[0-9]*\.?[0-9]+)")
CSHARP_ELAPSED_RE = re.compile(r"elapsed_ms=(?P<elapsed>-?[0-9]*\.?[0-9]+)")
CSHARP_CHECKSUM_RE = re.compile(r"checksum=(?P<checksum>-?[0-9]*\.?[0-9]+)")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="CUDA main.cu vs C# CPU spring-mass-damper benchmark comparator."
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
    parser.add_argument("--cuda-position", type=float, default=1.0)
    parser.add_argument("--cuda-velocity", type=float, default=0.0)
    parser.add_argument("--cuda-dt", type=float, default=0.016)
    parser.add_argument("--cuda-mass", type=float, default=1.0)
    parser.add_argument("--cuda-stiffness", type=float, default=120.0)
    parser.add_argument("--cuda-damping", type=float, default=0.2)
    parser.add_argument("--cuda-steps", type=int, default=8)

    parser.add_argument("--skip-csharp-build", action="store_true", help="Skip C# build")
    parser.add_argument(
        "--csharp-project",
        type=Path,
        default=repo_root / "cpu_csharp" / "CpuMassSpringDamper" / "CpuMassSpringDamper.csproj",
        help="C# project path",
    )
    parser.add_argument("--csharp-particles", type=int, default=8192)
    parser.add_argument("--csharp-steps", type=int, default=200)
    parser.add_argument("--csharp-substeps", type=int, default=1)
    parser.add_argument("--csharp-dt", type=float, default=0.016)
    parser.add_argument("--csharp-stiffness", type=float, default=120.0)
    parser.add_argument("--csharp-damping", type=float, default=0.2)
    parser.add_argument("--csharp-velocity-damping", type=float, default=0.999)
    parser.add_argument("--csharp-gravity-y", type=float, default=-9.81)
    parser.add_argument("--csharp-mass", type=float, default=1.0)
    parser.add_argument("--csharp-spacing", type=float, default=0.1)
    parser.add_argument("--csharp-fixed-first", type=str, default="true", choices=["true", "false", "1", "0"])

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


def parse_cuda_state(stdout: str) -> tuple[float, float]:
    for line in stdout.splitlines():
        m = CUDA_OUTPUT_RE.search(line)
        if m:
            return float(m.group("x")), float(m.group("v"))
    raise RuntimeError(f"Failed to parse CUDA output state.\nSTDOUT:\n{stdout}")


def parse_csharp_metrics(stdout: str) -> tuple[float, float]:
    elapsed = None
    checksum = None
    for line in stdout.splitlines():
        m_elapsed = CSHARP_ELAPSED_RE.search(line)
        if m_elapsed:
            elapsed = float(m_elapsed.group("elapsed"))
        m_checksum = CSHARP_CHECKSUM_RE.search(line)
        if m_checksum:
            checksum = float(m_checksum.group("checksum"))

    if elapsed is None or checksum is None:
        raise RuntimeError(f"Failed to parse C# benchmark output.\nSTDOUT:\n{stdout}")
    return elapsed, checksum


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


def run_cuda(args: argparse.Namespace, cuda_binary: Path) -> tuple[float, float, float]:
    command = [
        str(cuda_binary),
        str(args.cuda_position),
        str(args.cuda_velocity),
        str(args.cuda_dt),
        str(args.cuda_mass),
        str(args.cuda_stiffness),
        str(args.cuda_damping),
        str(args.cuda_steps),
    ]
    elapsed_ms, stdout = run_command(command, "CUDA run failed.")
    out_x, out_v = parse_cuda_state(stdout)
    return elapsed_ms, out_x, out_v


def run_csharp(args: argparse.Namespace) -> tuple[float, float, float]:
    project = args.csharp_project.resolve()
    command = [
        "dotnet",
        "run",
        "-c",
        "Release",
        "--project",
        str(project),
        "--",
        "--particles",
        str(args.csharp_particles),
        "--steps",
        str(args.csharp_steps),
        "--substeps",
        str(args.csharp_substeps),
        "--dt",
        str(args.csharp_dt),
        "--stiffness",
        str(args.csharp_stiffness),
        "--damping",
        str(args.csharp_damping),
        "--velocity-damping",
        str(args.csharp_velocity_damping),
        "--gravity-y",
        str(args.csharp_gravity_y),
        "--mass",
        str(args.csharp_mass),
        "--spacing",
        str(args.csharp_spacing),
        "--fixed-first",
        args.csharp_fixed_first,
    ]
    elapsed_ms, stdout = run_command(command, "C# run failed.")
    reported_elapsed_ms, checksum = parse_csharp_metrics(stdout)
    return elapsed_ms, reported_elapsed_ms, checksum


def save_csv(output_dir: Path, rows: list[dict[str, float]]) -> Path:
    csv_path = output_dir / "cuda_vs_csharp_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "cuda_wall_ms",
                "cuda_out_x",
                "cuda_out_v",
                "csharp_wall_ms",
                "csharp_reported_ms",
                "csharp_checksum",
                "wall_speedup_cuda_over_csharp",
                "reported_speedup_cuda_over_csharp",
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

    build_cuda(args)
    build_csharp(args)
    cuda_binary = resolve_cuda_binary(args.cuda_binary)

    for _ in range(args.warmup):
        run_cuda(args, cuda_binary)
        run_csharp(args)

    rows: list[dict[str, float]] = []
    cuda_wall_values: list[float] = []
    csharp_wall_values: list[float] = []
    csharp_reported_values: list[float] = []

    for i in range(1, args.samples + 1):
        cuda_wall_ms, cuda_x, cuda_v = run_cuda(args, cuda_binary)
        csharp_wall_ms, csharp_reported_ms, csharp_checksum = run_csharp(args)

        wall_ratio = cuda_wall_ms / csharp_wall_ms if csharp_wall_ms > 0.0 else 0.0
        reported_ratio = cuda_wall_ms / csharp_reported_ms if csharp_reported_ms > 0.0 else 0.0

        rows.append(
            {
                "sample": i,
                "cuda_wall_ms": cuda_wall_ms,
                "cuda_out_x": cuda_x,
                "cuda_out_v": cuda_v,
                "csharp_wall_ms": csharp_wall_ms,
                "csharp_reported_ms": csharp_reported_ms,
                "csharp_checksum": csharp_checksum,
                "wall_speedup_cuda_over_csharp": wall_ratio,
                "reported_speedup_cuda_over_csharp": reported_ratio,
            }
        )

        cuda_wall_values.append(cuda_wall_ms)
        csharp_wall_values.append(csharp_wall_ms)
        csharp_reported_values.append(csharp_reported_ms)

        print(
            f"sample {i}/{args.samples}: "
            f"cuda_wall={cuda_wall_ms:.3f}ms, "
            f"csharp_wall={csharp_wall_ms:.3f}ms, "
            f"csharp_reported={csharp_reported_ms:.3f}ms, "
            f"wall_ratio(cuda/csharp)={wall_ratio:.4f}"
        )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_csv(output_dir, rows)

    cuda_stats = summarize(cuda_wall_values)
    csharp_wall_stats = summarize(csharp_wall_values)
    csharp_reported_stats = summarize(csharp_reported_values)

    print("\n=== CUDA vs C# Benchmark Summary ===")
    print(f"samples={args.samples}, warmup={args.warmup}")
    print(
        "cuda wall(ms): "
        f"mean={cuda_stats['mean']:.3f}, median={cuda_stats['median']:.3f}, "
        f"min={cuda_stats['min']:.3f}, max={cuda_stats['max']:.3f}, stdev={cuda_stats['stdev']:.3f}"
    )
    print(
        "csharp wall(ms): "
        f"mean={csharp_wall_stats['mean']:.3f}, median={csharp_wall_stats['median']:.3f}, "
        f"min={csharp_wall_stats['min']:.3f}, max={csharp_wall_stats['max']:.3f}, stdev={csharp_wall_stats['stdev']:.3f}"
    )
    print(
        "csharp reported(ms): "
        f"mean={csharp_reported_stats['mean']:.3f}, median={csharp_reported_stats['median']:.3f}, "
        f"min={csharp_reported_stats['min']:.3f}, max={csharp_reported_stats['max']:.3f}, stdev={csharp_reported_stats['stdev']:.3f}"
    )
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
