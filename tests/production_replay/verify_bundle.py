"""CLI: run production-replay assertions against a debug bundle directory.

Usage:
    .venv/bin/python tests/production_replay/verify_bundle.py <bundle_dir>

Exits 0 if all assertions pass, 1 if any INVARIANT fails (other
severities log but don't exit nonzero — they're guidance, not gates).
"""

from __future__ import annotations

import sys
from pathlib import Path

from .assertions import (
    Bundle,
    SEVERITY_INVARIANT,
    format_report,
    run_all,
)


def main():
    if len(sys.argv) < 2:
        print("usage: verify_bundle.py <bundle_dir>")
        sys.exit(2)
    bundle = Bundle.from_path(Path(sys.argv[1]))
    results = run_all(bundle)
    print(format_report(bundle, results))
    invariant_fails = [r for r in results
                       if r.severity == SEVERITY_INVARIANT and not r.passed]
    sys.exit(1 if invariant_fails else 0)


if __name__ == "__main__":
    main()
