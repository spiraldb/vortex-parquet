#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the benchmark's exit status and JSONL acceptance result with controlled timings."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import benchmark


class AcceptanceTest(unittest.TestCase):
    def test_complete_suite_required(self):
        cases = [
            # One loss must fail even when the other queries make the aggregate much faster.
            ("all win", [], 0.5, False, False, 100, 0),
            ("one loss", [], 1.01, False, False, 100, 1),
            ("one tie", [], 1.0, False, False, 100, 1),
            ("invalid timing", [], float("nan"), False, False, 100, 1),
            ("wrong answer", [], 0.5, True, False, 100, 1),
            ("execution error", [], 0.5, False, True, 100, 1),
            ("selected query", ["--queries", "1"], 0.5, False, False, 100, 1),
            ("weaker baseline", ["--no-native-multifile"], 0.5, False, False, 100, 1),
        ]
        for suite, count in [("tpch", 22), ("clickbench", 43)]:
            suite_cases = cases + ([
                ("preflight", ["--max-files", "100"], 0.5, False, False, 100, 1),
                ("missing shard", [], 0.5, False, False, 99, 1),
            ] if suite == "clickbench" else [])
            for name, options, last_ratio, wrong_answer, error, shards, expected_exit in suite_cases:
                with self.subTest(suite=suite, case=name), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    names = ([f"hits_{i}.parquet" for i in range(shards)] if suite == "clickbench" else
                             [f"{table}.parquet" for table in
                              ("lineitem", "orders", "part", "partsupp", "supplier", "customer", "nation", "region")])
                    for filename in names:
                        (root / filename).touch()
                    extension = root / "test.duckdb_extension"
                    extension.touch()
                    output = root / "result.jsonl"
                    connections = [MagicMock(), MagicMock()]
                    for connection in connections:
                        connection.execute.return_value.fetchall.return_value = [(1, 1)]
                    calls = 0

                    def measured(connection, sql, timeout):
                        nonlocal calls
                        last = calls // 6 == count - 1  # Three paired samples per query.
                        calls += 1
                        vortex = connection is connections[1]
                        if last and vortex and error:
                            raise RuntimeError("query failed")
                        rows = [(2 if last and vortex and wrong_answer else 1,)]
                        seconds = (last_ratio if last else 0.5) if vortex else 1.0
                        return seconds, rows

                    argv = ["benchmark.py", "--suite", suite, "--extension", str(extension),
                            "--data", str(root), "--output", str(output), "--warmup", "0",
                            "--repeat", "3", *options]
                    with patch.object(sys, "argv", argv), \
                         patch.object(benchmark.duckdb, "connect", side_effect=connections), \
                         patch.object(benchmark, "execute", side_effect=measured), \
                         contextlib.redirect_stdout(io.StringIO()):
                        exit_code = 0
                        try:
                            benchmark.main()
                        except SystemExit as stopped:
                            exit_code = stopped.code
                    self.assertEqual(exit_code, expected_exit)
                    records = [json.loads(line) for line in output.read_text().splitlines()]
                    summary = records[-1]
                    self.assertEqual(summary["kind"], "summary")
                    self.assertEqual(summary["suite_passed"], expected_exit == 0)
                    self.assertEqual(summary["expected_queries"], count)
                    self.assertEqual(summary["queries"], 1 if name == "selected query" else count)
                    if name == "selected query":
                        self.assertFalse(summary["all_queries_faster"])
                        self.assertEqual(summary["missing_queries"], list(range(2, count + 1)))
                    if name in ("one loss", "one tie"):
                        self.assertLess(summary["vortex_over_native_geomean"], 1)
                        self.assertEqual([q["query"] for q in summary["performance_failures"]], [count])
                    if name in ("selected query", "weaker baseline", "preflight", "missing shard"):
                        self.assertTrue(summary["coverage_failures"])


if __name__ == "__main__":
    unittest.main()
