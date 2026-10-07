"""Run and plot the local eight-core bounding-box benchmark."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))

def main() -> None:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "8")
    for stage in ("reference", "native", "merge"):
        subprocess.run(
            [sys.executable, "-m", "experiments.toy.run_bbox_full",
             "--output-dir", str(HERE), "--stage", stage],
            cwd=ROOT / "code", env=env, check=True,
        )


if __name__ == "__main__":
    main()
