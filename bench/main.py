from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)


class MattressModel:
    """Parser for mattress.log model files"""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.particles: list[dict] = []
        self.particle_count = 0
        self.spring_count = 0
        self.gravity = (0.0, -9.81, 0.0)
        self.dt = 0.001
        self.stiffness = 500.0
        self._parse()

    def _parse(self) -> None:
        """Parse mattress.log file format"""
        with open(self.log_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header line
        if not lines:
            raise ValueError("Empty log file")
        
        header = lines[0]
        if "first-frame debug snapshot" not in header:
            raise ValueError("Invalid mattress.log format")
        
        # Parse parameters line
        params_line = lines[1]
        # Format: gravity=(0.00, -9.81, 0.00), dt=0.0010, stiffness=500.000
        grav_match = re.search(r'gravity=\(([^,]+),\s*([^,]+),\s*([^)]+)\)', params_line)
        if grav_match:
            self.gravity = tuple(float(x) for x in grav_match.groups())
        dt_match = re.search(r'dt=([0-9.]+)', params_line)
        if dt_match:
            self.dt = float(dt_match.group(1))
        stiff_match = re.search(r'stiffness=([0-9.]+)', params_line)
        if stiff_match:
            self.stiffness = float(stiff_match.group(1))
        
        # Parse counts line
        counts_line = lines[2]
        # Format: particles=632, springs=1890
        particle_match = re.search(r'particles=(\d+)', counts_line)
        spring_match = re.search(r'springs=(\d+)', counts_line)
        if particle_match:
            self.particle_count = int(particle_match.group(1))
        if spring_match:
            self.spring_count = int(spring_match.group(1))
        
        # Parse particle lines
        # Format: P[0] pos=(-5.00, 5.75, -5.00) vel=(0.00, 0.00, 0.00) gravity=(0.00, -9.81, 0.00) force=(0.00, -9.81, 0.00) fixed=False
        p_pattern = re.compile(
            r'P\[(\d+)\]\s+pos=\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
            r'\s+vel=\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
            r'.*fixed=(\w+)'
        )
        for line in lines[3:]:
            match = p_pattern.search(line)
            if match:
                idx, px, py, pz, vx, vy, vz, fixed = match.groups()
                self.particles.append({
                    'idx': int(idx),
                    'pos': (float(px), float(py), float(pz)),
                    'vel': (float(vx), float(vy), float(vz)),
                    'fixed': fixed.lower() == 'true'
                })

    def get_1d_equivalent_params(self) -> dict[str, float]:
        """Extract 1D equivalent parameters from 3D mattress model"""
        # Calculate effective mass from particle count
        total_particles = len(self.particles)
        free_particles = sum(1 for p in self.particles if not p['fixed'])
        
        # Use average effective mass
        effective_mass = 1.0 if free_particles == 0 else float(total_particles) / float(free_particles)
        
        # Use system stiffness directly
        effective_stiffness = self.stiffness
        
        # Damping scaled by particle count
        effective_damping = 0.2 * (float(self.spring_count) / 1890.0)  # Normalize to mattress baseline
        
        return {
            'position': 0.0,
            'velocity': 0.0,
            'mass': effective_mass,
            'stiffness': effective_stiffness,
            'damping': effective_damping,
            'dt': self.dt,
        }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run CUDA and C# benchmarks with the same 1D spring-mass-damper I/O spec."
    )

    parser.add_argument("--samples", type=int, default=10, help="Number of measured runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs before sampling")
    parser.add_argument("--model", type=str, default=None, help="Model to load (e.g., 'mattress')")
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
    parser.add_argument(
        "--run",
        type=str,
        default="both",
        choices=["cuda", "csharp", "both"],
        help="Which implementation to run (default: both)"
    )
    return parser.parse_args()


def run_command(command: list[str], fail_prefix: str) -> tuple[float, str]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
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
    is_windows = sys.platform.startswith("win")
    shared_lib_suffixes = {".dll", ".so", ".dylib"}

    candidates: list[Path] = [resolved]

    # If a shared library path is provided, try likely executable names in same folder.
    if resolved.suffix.lower() in shared_lib_suffixes:
        candidates.append(resolved.parent / ("main.exe" if is_windows else "main"))
        candidates.append(resolved.parent / ("mass_spring_native.exe" if is_windows else "mass_spring_native"))

    fallback_dir = resolved.parent / "Release"
    if is_windows:
        if resolved.suffix:
            candidates.append(fallback_dir / f"{resolved.stem}.exe")
        else:
            candidates.append(fallback_dir / f"{resolved.name}.exe")
    candidates.append(fallback_dir / resolved.name)
    candidates.append(fallback_dir / ("main.exe" if is_windows else "main"))

    checked: list[Path] = []
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() not in shared_lib_suffixes:
            return candidate

    raise FileNotFoundError(
        "CUDA executable not found. "
        f"Provided path: {resolved}. Checked candidates: {', '.join(str(p) for p in checked)}"
    )


def build_cuda(args: argparse.Namespace) -> None:
    if args.skip_cuda_build:
        return

    # Windows environments often do not have `bash` in PATH.
    # In that case, configure/build directly with CMake.
    if sys.platform.startswith("win"):
        repo_root = Path(__file__).resolve().parents[1]
        cuda_dir = repo_root / "cuda"
        build_dir = cuda_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                "cmake",
                "-S",
                str(cuda_dir),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DMSS_CUDA_ARCHITECTURES={args.cuda_arch}",
                "-DMSS_ENABLE_LINEINFO=ON",
            ],
            "CUDA CMake configure failed.",
        )
        run_command(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "main",
            ],
            "CUDA CMake build failed.",
        )
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


def apply_model_params(args: argparse.Namespace, model_name: str) -> None:
    """Load model and override benchmark parameters"""
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "bench" / "models" / f"{model_name}.log"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = MattressModel(model_path)
    params = model.get_1d_equivalent_params()
    
    # Override arguments with model parameters
    args.position = params['position']
    args.velocity = params['velocity']
    args.mass = params['mass']
    args.stiffness = params['stiffness']
    args.damping = params['damping']
    args.dt = params['dt']
    
    print(f"Model loaded: {model.particle_count} particles, {model.spring_count} springs")
    print(f"1D equivalent parameters:")
    print(f"  mass={args.mass:.6f}, stiffness={args.stiffness:.6f}, damping={args.damping:.6f}")
    print(f"  dt={args.dt:.6f}\n")


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.warmup < 0:
        raise ValueError("samples must be >= 1 and warmup must be >= 0")
    if args.steps < 1:
        raise ValueError("steps must be >= 1")
    
    # Load model if specified
    if args.model:
        apply_model_params(args, args.model)

    # Only build the selected implementations
    cuda_binary = None
    if args.run in ("cuda", "both"):
        build_cuda(args)
        cuda_binary = resolve_cuda_binary(args.cuda_binary)

    if args.run in ("csharp", "both"):
        build_csharp(args)

    for _ in range(args.warmup):
        if args.run in ("cuda", "both"):
            run_cuda(args, cuda_binary)
        if args.run in ("csharp", "both"):
            run_csharp(args)

    rows: list[dict[str, float]] = []
    cuda_elapsed_values: list[float] = []
    csharp_elapsed_values: list[float] = []
    cuda_wall_values: list[float] = []
    csharp_wall_values: list[float] = []
    diff_checksum_values: list[float] = []

    for i in range(1, args.samples + 1):
        # Initialize default metrics so code can handle skipped runs
        cuda_wall_ms = 0.0
        csharp_wall_ms = 0.0
        cuda_metrics = {"elapsed_ms": 0.0, "output_x": 0.0, "output_v": 0.0, "checksum": 0.0}
        csharp_metrics = {"elapsed_ms": 0.0, "output_x": 0.0, "output_v": 0.0, "checksum": 0.0}

        if args.run in ("cuda", "both"):
            cuda_wall_ms, cuda_metrics = run_cuda(args, cuda_binary)

        if args.run in ("csharp", "both"):
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
