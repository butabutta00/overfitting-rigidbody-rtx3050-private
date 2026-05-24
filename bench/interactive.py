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
from typing import Iterable, Iterator

OUTPUT_RE = re.compile(r"Output\s*->\s*x=(?P<x>-?[0-9]*\.?[0-9]+)\s*,\s*v=(?P<v>-?[0-9]*\.?[0-9]+)")


def parse_output_line(line: str) -> tuple[float, float] | None:
    m = OUTPUT_RE.search(line)
    if not m:
        return None
    return float(m.group("x")), float(m.group("v"))


def generate_sequence(initial_x: float, initial_v: float, dt: float, mass: float, stiffness: float, damping: float, steps: int) -> Iterator[str]:
    """Yield parameter lines to send to the binary.

    Each yielded line is: position velocity dt mass stiffness damping steps
    Modify this generator to sweep parameters or apply time-varying forces.
    """
    x = initial_x
    v = initial_v
    for i in range(steps):
        # Example: simple sequence that keeps same params but feeds new state each timestep.
        yield f"{x} {v} {dt} {mass} {stiffness} {damping} 1\n"
        # The harness will update x,v from the binary output; here we just yield current state.


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
    args = parser.parse_args()

    seq = generate_sequence(args.x0, args.v0, args.dt, args.mass, args.stiffness, args.damping, args.steps)

    for sample, x, v in run_interactive(args.binary, seq):
        print(f"step {sample}: x={x:.6f}, v={v:.6f}")


if __name__ == "__main__":
    main()
