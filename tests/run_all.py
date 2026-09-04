#!/usr/bin/env python3
"""Run every test. `python3 tests/run_all.py`

    test_geometry.py    the pose maths, no images, milliseconds
    test_workflows.py   the scripts end to end on the repo's images, ~3 minutes
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["test_geometry.py", "test_workflows.py"]


def main():
    only = sys.argv[1:]
    failures = []
    for suite in SUITES:
        if only and not any(o in suite for o in only):
            continue
        print(f"\n=== {suite} ===")
        code = subprocess.call([sys.executable, os.path.join(HERE, suite)])
        if code != 0:
            failures.append(suite)
    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all suites passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
