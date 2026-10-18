"""Print how many tests the suite collects (used in the README, checked in CI).

`-o addopts=` clears any `addopts = "-q"` in pyproject.toml. Without it,
pytest runs at -qq, prints only per-file counts, and the total line this
script reads is missing. Per-file counts are also summed as a fallback, and
the script exits non-zero if neither works, so a broken count fails CI
instead of printing 0.
"""
import re
import subprocess
import sys

proc = subprocess.run(
    [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider"],
    capture_output=True, text=True,
)
out = proc.stdout
m = re.search(r"(\d+) tests? collected", out)
if m:
    n = int(m.group(1))
else:
    per_file = re.findall(r"^\S+\.py: (\d+)$", out, flags=re.M)
    if not per_file:
        sys.exit(f"could not read a test count from pytest output (exit {proc.returncode})")
    n = sum(int(x) for x in per_file)
print(f"{n} tests collected")
