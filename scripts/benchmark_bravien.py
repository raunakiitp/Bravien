#!/usr/bin/env python3
"""Bravien Intelligence Benchmark Runner CLI (§Phase 13).

Executes the modular 51-case evaluation suite, gathers telemetry,
computes category scores and security audits, and writes reports to:
- reports/phase13/latest.json
- reports/phase13/latest.md
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    cmd = ["npx", "tsx", "scripts/run_benchmark.ts"]
    result = subprocess.run(cmd, cwd=str(ROOT), shell=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
