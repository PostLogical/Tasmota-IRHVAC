"""Production-bundle replay-and-assert framework.

Encodes expected behavior of a deployed production build as assertions that
can be evaluated against a captured debug bundle (`tick_log.jsonl` etc.).

Usage:
    .venv/bin/python -m tests.production_replay.verify_bundle <bundle_dir>

Or programmatically:
    from tests.production_replay.assertions import Bundle, run_all, format_report
    bundle = Bundle.from_path("local/debug_bundles/...")
    results = run_all(bundle)
    print(format_report(bundle, results))
"""
