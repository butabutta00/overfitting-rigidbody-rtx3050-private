"""Interactive harness to drive the CUDA mass-spring binary continuously.

Usage example:
    python bench/interactive.py --binary ../cuda/build/main --steps 100

This will spawn the binary with `--interactive`, send 100 sequential steps (one per line)
and print the outputs. You can adapt `generate_sequence()` to test arbitrary parameter sweeps
or time-varying external forces (by modifying the mass/spring/damping or state values).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, Callable, Optional
import math
import random

OUTPUT_RE = re.compile(r"Output\s*->\s*x=(?P<x>-?[0-9]*\.?[0-9]+)\s*,\s*v=(?P<v>-?[0-9]*\.?[0-9]+)")


def parse_output_line(line: str) -> tuple[float, float] | None:
    m = OUTPUT_RE.search(line)
    if not m:
        return None
    return float(m.group("x")), float(m.group("v"))


def strategy_hold(step: int, last_x: float, last_v: float, dt: float, mass: float, stiffness: float, damping: float) -> str:
    return f"{last_x} {last_v} {dt} {mass} {stiffness} {damping} 1\n"


def strategy_sine_k(step: int, last_x: float, last_v: float, dt: float, mass: float, stiffness: float, damping: float, amp: float = 0.2, freq: float = 1.0) -> str:
    k = stiffness * (1.0 + amp * math.sin(2.0 * math.pi * freq * step))
    return f"{last_x} {last_v} {dt} {mass} {k} {damping} 1\n"


def strategy_noise_k(step: int, last_x: float, last_v: float, dt: float, mass: float, stiffness: float, damping: float, scale: float = 0.1) -> str:
    k = stiffness * (1.0 + scale * (2.0 * random.random() - 1.0))
    return f"{last_x} {last_v} {dt} {mass} {k} {damping} 1\n"


def strategy_grid_k(step: int, last_x: float, last_v: float, dt: float, mass: float, stiffness: float, damping: float, grid_values: list[float] | None = None) -> str:
    if not grid_values:
        k = stiffness
    else:
        k = grid_values[step % len(grid_values)]
    return f"{last_x} {last_v} {dt} {mass} {k} {damping} 1\n"


def run_loop_strategy(
    binary: Path,
    steps: int,
    x0: float,
    v0: float,
    dt: float,
    mass: float,
    stiffness: float,
    damping: float,
    strategy: Callable[..., str],
    strategy_kwargs: Optional[dict] = None,
):
    cmd = [str(binary), "--interactive"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("Failed to open subprocess pipes")

    last_x = x0
    last_v = v0
    sample = 0
    strategy_kwargs = strategy_kwargs or {}

    try:
        for step in range(steps):
            sample += 1
            line = strategy(step, last_x, last_v, dt, mass, stiffness, damping, **strategy_kwargs)
            proc.stdin.write(line)
            proc.stdin.flush()

            # Read stdout lines until we find an Output line
            while True:
                out_line = proc.stdout.readline()
                if out_line == "":
                    # EOF
                    raise RuntimeError("Subprocess terminated unexpectedly")
                parsed = parse_output_line(out_line)
                if parsed is not None:
                    last_x, last_v = parsed
                    yield sample, last_x, last_v
                    break
                # otherwise keep reading (could be Input or debug text)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()
        proc.wait(timeout=1)


def run_interactive(binary: Path, lines: Iterable[str]) -> Iterator[tuple[int, float, float]]:
    cmd = [str(binary), "--interactive"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("Failed to open subprocess pipes")

    sample = 0
    try:
        for line in lines:
            sample += 1
            proc.stdin.write(line)
            proc.stdin.flush()

            # Read stdout lines until we find an Output line
            while True:
                out_line = proc.stdout.readline()
                if out_line == "":
                    # EOF
                    raise RuntimeError("Subprocess terminated unexpectedly")
                parsed = parse_output_line(out_line)
                if parsed is not None:
                    x, v = parsed
                    yield sample, x, v
                    break
                # otherwise keep reading (could be Input or debug text)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()
        proc.wait(timeout=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive CUDA mass-spring binary interactively")
    parser.add_argument("--binary", type=Path, required=True, help="Path to CUDA binary built with --interactive support")
    parser.add_argument("--x0", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.016)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--stiffness", type=float, default=120.0)
    parser.add_argument("--damping", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=10, help="Number of interactive steps to run")
    parser.add_argument("--mode", type=str, default="hold", choices=["hold", "sine_k", "noise_k", "grid_k"], help="Driving mode for parameters")
    parser.add_argument("--amp", type=float, default=0.2, help="Amplitude for sine_k mode")
    parser.add_argument("--freq", type=float, default=1.0, help="Frequency (Hz) for sine_k mode")
    parser.add_argument("--scale", type=float, default=0.1, help="Scale for noise_k mode")
    parser.add_argument("--grid-start", type=float, default=60.0, help="Grid start k for grid_k mode")
    parser.add_argument("--grid-end", type=float, default=180.0, help="Grid end k for grid_k mode")
    parser.add_argument("--grid-steps", type=int, default=6, help="Number of grid points for grid_k mode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for noise_k")
    args = parser.parse_args()

    strategy = strategy_hold
    strategy_kwargs = {}
    if args.mode == "hold":
        strategy = strategy_hold
    elif args.mode == "sine_k":
        strategy = strategy_sine_k
        strategy_kwargs = {"amp": args.amp, "freq": args.freq}
    elif args.mode == "noise_k":
        strategy = strategy_noise_k
        strategy_kwargs = {"scale": args.scale}
        if args.seed is not None:
            random.seed(args.seed)
    elif args.mode == "grid_k":
        grid_values = [args.grid_start + (args.grid_end - args.grid_start) * i / max(1, args.grid_steps - 1) for i in range(args.grid_steps)]
        strategy = strategy_grid_k
        strategy_kwargs = {"grid_values": grid_values}

    for sample, x, v in run_loop_strategy(
        args.binary,
        args.steps,
        args.x0,
        args.v0,
        args.dt,
        args.mass,
        args.stiffness,
        args.damping,
        strategy,
        strategy_kwargs,
    ):
        print(f"step {sample}: x={x:.6f}, v={v:.6f}")


if __name__ == "__main__":
    main()
