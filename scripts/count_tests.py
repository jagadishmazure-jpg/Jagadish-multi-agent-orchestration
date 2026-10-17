"""Print how many tests the suite collects (used in the README, checked in CI)."""
import re
import subprocess
import sys

out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                     capture_output=True, text=True).stdout
m = re.search(r"(\d+) tests? collected", out)
print(f"{m.group(1) if m else 0} tests collected")
