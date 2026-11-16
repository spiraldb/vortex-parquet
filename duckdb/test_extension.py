#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the packaged extension's entrypoint and C ABI in an actual DuckDB process."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def limit_open_files():
    # Bound only this child process, so leaking an unexecuted scan fails quickly and cannot
    # change resource limits for the test runner or other tests.
    import resource
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = 128 if hard == resource.RLIM_INFINITY else min(128, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("--duckdb", default="duckdb")
    args = parser.parse_args()
    extension = str(args.extension.resolve()).replace("'", "''")
    # Resolve an explicitly provided relative executable before changing the child cwd.
    executable = str(Path(args.duckdb).resolve()) if "/" in args.duckdb else args.duckdb
    # Binding and discarding a scan must release its file without requiring a data scan to
    # drive cleanup. The final query also proves real I/O still works after the planning loop.
    planning_checks = "\n".join([
        "PREPARE unused_scan AS SELECT sum(i) FROM read_parquet_vortex('input.parquet');",
        "DEALLOCATE unused_scan;",
    ] * 192)
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.check_output(
            [executable, "-no-agent", "-unsigned", "-batch", "-bail", "-csv", "-noheader"],
            cwd=directory,
            input=f"""
                LOAD '{extension}';
                COPY (SELECT i, 'row-' || i AS label FROM range(5003) t(i))
                    TO 'input.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE 2048);
                {planning_checks}
                SELECT count(*), sum(i)::BIGINT, min(label)
                    FROM read_parquet_vortex('input.parquet');
            """,
            text=True,
            preexec_fn=limit_open_files if os.name == "posix" else None,
        )
    expected = "5003,12512503,row-0"
    if result.strip() != expected:
        raise SystemExit(f"unexpected extension result: {result!r}; expected {expected!r}")
    print("DuckDB loadable extension smoke test passed")


if __name__ == "__main__":
    main()
