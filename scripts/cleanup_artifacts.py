#!/usr/bin/env python3
"""Remove generated environment and cache artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_TARGETS = [
    "env_artifacts",
    "env_artifacts_stream",
    "env_artifacts_no_csv",
    "mlx_cache",
]


def remove_path(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        print(f"Would remove {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"Removed {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for item in args.paths:
        remove_path(Path(item), dry_run=args.dry_run)

if __name__ == "__main__":
    main()
