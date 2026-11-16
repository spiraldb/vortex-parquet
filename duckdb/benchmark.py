#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Require wins on all 22 TPC-H or all 43 ClickBench queries against DuckDB read_parquet."""

import argparse
import datetime
import decimal
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import threading
import time

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def normalized(value):
    if isinstance(value, float):
        return ["float", format(value, ".7g")]
    if isinstance(value, (datetime.date, datetime.datetime, decimal.Decimal)):
        return [type(value).__name__, str(value)]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    return [type(value).__name__, value]


def row_key(row):
    return json.dumps([normalized(value) for value in row], ensure_ascii=False)


def answers_equal(left, right):
    if len(left) != len(right):
        return False
    for a, b in zip(sorted(left, key=row_key), sorted(right, key=row_key)):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if isinstance(x, float) and isinstance(y, float):
                if math.isnan(x) and math.isnan(y):
                    continue
                if not math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-8):
                    return False
            elif type(x) is not type(y) or x != y:
                return False
    return True


def execute(connection, sql, timeout):
    timer = threading.Timer(timeout, connection.interrupt)
    timer.start()
    start = time.perf_counter()
    try:
        rows = connection.execute(sql).fetchall()
        return time.perf_counter() - start, rows
    finally:
        timer.cancel()
        timer.join()


def queries(suite):
    if suite == "tpch":
        return [(i, (ROOT / f"benchmarks/tpc-sql/tpch/q{i}.sql").read_text().strip().rstrip(";"))
                for i in range(1, 23)]
    # The repository's ClickBench queries use standard timestamps. DuckDB spells byte length
    # STRLEN; its OCTET_LENGTH accepts only BLOB. Both arms receive the same translated SQL.
    lines = (ROOT / "benchmarks/sql/clickbench.sql").read_text().splitlines()
    return [(i, line.split(";", 1)[0].replace("OCTET_LENGTH", "STRLEN"))
            for i, line in enumerate(lines, 1) if line.strip()]


def deterministic_limit(connection, sql):
    """Resolve ties at the LIMIT boundary, so different scheduling cannot change the answers."""
    match = re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE)
    if not match:
        return sql
    prefix, suffix = sql[:match.start()].rstrip(), sql[match.start():]
    columns = len(connection.execute(f"DESCRIBE ({sql})").fetchall())
    keys = ", ".join(str(i) for i in range(1, columns + 1))
    # The suite's LIMIT clauses are outermost. All of their outer ORDER BY clauses follow the
    # last closing parenthesis only in some queries, so inspect the final query block's text.
    if re.search(r"\bORDER\s+BY\b", prefix, re.IGNORECASE):
        prefix += ", " + keys
    else:
        prefix += " ORDER BY " + keys
    return prefix + " " + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--suite", choices=["tpch", "clickbench"], required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="TPC-H table directory, ClickBench file, or directory of hits_*.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--queries", help="Comma-separated one-based query numbers")
    parser.add_argument("--max-files", type=int, help="Preflight only: limit ClickBench input shards")
    parser.add_argument("--native-multifile", action=argparse.BooleanOptionalAction, default=True,
                        help="Let DuckDB read each table's files in one native scan instead of UNION ALL")
    parser.add_argument("--vortex-multifile", action=argparse.BooleanOptionalAction, default=True,
                        help="Let Vortex read each table's files in one scan instead of UNION ALL")
    args = parser.parse_args()
    if args.threads < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("threads/repeat must be positive and warmup nonnegative")
    if args.max_files is not None and (args.suite != "clickbench" or args.max_files < 1):
        parser.error("max-files must be positive and is only supported for ClickBench")
    expected_ids = set(range(1, (22 if args.suite == "tpch" else 43) + 1))
    suite_queries = queries(args.suite)
    if len(suite_queries) != len(expected_ids) or {i for i, _ in suite_queries} != expected_ids:
        parser.error(f"{args.suite} must contain exactly queries 1 through {len(expected_ids)}")
    try:
        selected = set(map(int, args.queries.split(","))) if args.queries is not None else expected_ids
    except ValueError:
        parser.error("queries must be comma-separated integers")
    if not selected or not selected <= expected_ids:
        parser.error(f"queries must be between 1 and {len(expected_ids)}")
    extension = args.extension.resolve()
    data = args.data.resolve()
    if args.suite == "tpch":
        tables = {p.stem: [p] for p in sorted(data.glob("*.parquet"))}
        required = {"lineitem", "orders", "part", "partsupp", "supplier", "customer", "nation", "region"}
        if set(tables) != required:
            parser.error(f"expected the eight TPC-H tables, got {sorted(tables)}")
    else:
        files = [data] if data.is_file() else sorted(data.glob("hits_*.parquet"))
        if args.max_files:
            files = files[:args.max_files]
        tables = {"hits": files}
        if not files:
            parser.error("no ClickBench parquet files found")
    missing_queries = sorted(expected_ids - selected)
    coverage_failures = []
    if missing_queries:
        coverage_failures.append(f"unmeasured queries: {missing_queries}")
    if args.max_files is not None:
        coverage_failures.append("max-files is a diagnostic preflight, not a complete run")
    if args.suite == "clickbench" and len(tables["hits"]) != 100:
        coverage_failures.append("full ClickBench requires all 100 input shards")
    if not args.native_multifile:
        coverage_failures.append("acceptance requires DuckDB read_parquet file-list scans")
    connections = {}
    # The historical mode isolates per-file readers with matching UNION ALL views. The native
    # multi-file mode additionally measures each reader's file-list planning and scheduling.
    for arm, function in [("native", "read_parquet"), ("vortex", "read_parquet_vortex")]:
        connection = duckdb.connect(config={"allow_unsigned_extensions": "true", "threads": str(args.threads)})
        connection.execute(f"LOAD {literal(extension)}")
        for table, files in tables.items():
            multifile = args.native_multifile if arm == "native" else args.vortex_multifile
            if multifile and len(files) > 1:
                paths = ", ".join(literal(p) for p in files)
                scan = f"SELECT * FROM {function}([{paths}])"
            else:
                scan = " UNION ALL ".join(f"SELECT * FROM {function}({literal(p)})" for p in files)
            connection.execute(f'CREATE VIEW "{table}" AS {scan}')
        connections[arm] = connection
    native = connections["native"]
    workload = [(i, deterministic_limit(native, sql)) for i, sql in suite_queries if i in selected]
    files = [p for paths in tables.values() for p in paths]
    metadata = {
        "kind": "run", "suite": args.suite, "duckdb": duckdb.__version__,
        "arms": {"native": "DuckDB read_parquet", "vortex": "DuckDB read_parquet_vortex"},
        "storage": "Both arms query the same Parquet files through views; no imported DuckDB tables",
        "native_multifile": args.native_multifile,
        "vortex_multifile": args.vortex_multifile,
        "platform": platform.platform(), "threads": args.threads,
        "logical_cpus": os.cpu_count(),
        "load_average_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "reader_environment": {key: value for key, value in sorted(os.environ.items())
                               if key.startswith("VORTEX_PARQUET_")},
        "warmup": args.warmup, "repeat": args.repeat, "timeout": args.timeout,
        "query_selection": args.queries, "max_files": args.max_files,
        "performance_requirement": "All 22 TPC-H or all 43 ClickBench queries must return correct answers "
                                   "and have read_parquet_vortex/read_parquet median < 1 on full inputs against read_parquet file-list scans",
        "expected_queries": len(expected_ids), "missing_queries": missing_queries,
        "coverage_failures": coverage_failures,
        "extension": str(extension), "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "inputs": [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                    "metadata": native.execute("SELECT num_rows,num_row_groups FROM parquet_file_metadata(?)", [str(p)]).fetchall()}
                   for p in files],
        "queries": workload, "timing": "prepare + execute + fetchall; comparison outside timer",
        "cache": "warm OS cache; engine defaults; fresh scan for each execution",
        "ties": "LIMIT queries receive output-column tie breakers, identical on both arms",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ratios, failures, performance_failures = [], [], []
    with args.output.open("x") as output:
        def emit(record):
            output.write(json.dumps(record, default=str) + "\n")
            output.flush()
        emit(metadata)
        for query_id, sql in workload:
            reference = None
            samples = {arm: [] for arm in connections}
            failed = False
            for iteration in range(args.warmup + args.repeat):
                order = ["native", "vortex"] if (iteration + query_id) % 2 == 0 else ["vortex", "native"]
                for arm in order:
                    try:
                        seconds, rows = execute(connections[arm], sql, args.timeout)
                        if not math.isfinite(seconds) or seconds <= 0:
                            raise ValueError(f"invalid execution time: {seconds}")
                        if reference is None:
                            reference = rows
                        if not answers_equal(reference, rows):
                            raise ValueError(f"answer mismatch: reference={reference[:10]!r}, {arm}={rows[:10]!r}")
                        digest = hashlib.sha256("\n".join(sorted(map(row_key, rows))).encode()).hexdigest()
                        emit({"kind": "sample", "query": query_id, "arm": arm, "iteration": iteration,
                              "warmup": iteration < args.warmup, "seconds": seconds, "rows": len(rows), "digest": digest})
                        if iteration >= args.warmup:
                            samples[arm].append(seconds)
                    except Exception as error:
                        emit({"kind": "failure", "query": query_id, "arm": arm, "error": str(error)})
                        failures.append(query_id)
                        print(f"q{query_id}: FAILED ({arm}): {error}", flush=True)
                        failed = True
                        break
                if failed:
                    break
            if not failed:
                medians = {arm: statistics.median(values) for arm, values in samples.items()}
                ratio = medians["vortex"] / medians["native"]
                ratios.append(ratio)
                if not math.isfinite(ratio) or not 0 < ratio < 1:
                    performance_failures.append({"query": query_id, "ratio": ratio,
                                                 "medians": medians})
                emit({"kind": "query", "query": query_id, "medians": medians, "ratio": ratio,
                      "spread": {arm: max(values) / min(values) for arm, values in samples.items()}})
                print(f"q{query_id}: read_parquet={medians['native']:.4f}s read_parquet_vortex={medians['vortex']:.4f}s ratio={ratio:.3f}", flush=True)
        geomean = (math.exp(statistics.mean(map(math.log, ratios)))
                   if not failures and all(math.isfinite(r) and r > 0 for r in ratios) else None)
        all_queries_faster = len(ratios) == len(expected_ids) and not failures and not performance_failures
        suite_passed = all_queries_faster and not coverage_failures
        summary = {"kind": "summary", "queries": len(workload), "passed": len(ratios),
                   "expected_queries": len(expected_ids), "missing_queries": missing_queries,
                   "coverage_failures": coverage_failures, "suite_passed": suite_passed,
                   "failures": failures, "vortex_over_native_geomean": geomean,
                   "performance_passed": len(ratios) - len(performance_failures),
                   "performance_failures": performance_failures,
                   "all_queries_faster": all_queries_faster,
                   "load_average_end": os.getloadavg() if hasattr(os, "getloadavg") else None}
        emit(summary)
        print(json.dumps(summary), flush=True)
    for connection in connections.values():
        connection.close()
    # A suite-level geometric mean cannot compensate for an individual query that ties or loses.
    # Keep the complete measurements above even when this strict performance gate fails.
    if not suite_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
