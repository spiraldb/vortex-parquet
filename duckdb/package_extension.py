#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package the cdylib with DuckDB's extension footer (no network dependencies)."""

import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--duckdb", default="duckdb", help="DuckDB 2.0.0 CLI for version/platform detection")
    args = parser.parse_args()
    version, platform = subprocess.check_output(
        [args.duckdb, "-no-agent", "-csv", "-noheader", "-c", "SELECT version(); PRAGMA platform;"],
        text=True,
    ).strip().splitlines()
    # The C++ bridge targets an exact engine version, independently of the stable V1 C API.
    if version != "v2.0.0":
        parser.error(f"this extension targets DuckDB v2.0.0, got {version}")
    output = args.library.with_name("vortex_parquet.duckdb_extension")
    if args.library.resolve() == output.resolve():
        parser.error("input must be the raw cdylib, not an already packaged extension")
    # Format specified by duckdb/extension-ci-tools/scripts/append_extension_metadata.py:
    # eight reversed 32-byte fields, then a 256-byte unsigned signature, in a custom section.
    # DuckDB 2.0 retains the V1 C API at v1.5.6. Unstable metadata selects the V2 loader.
    # The entrypoint separately checks the exact runtime before reading the API table.
    fields = ["4", platform, "v1.5.6", "0.1.0", "C_STRUCT", "", "", ""]
    footer = b"\x00\x93\x04\x10duckdb_signature\x80\x04"
    for field in reversed(fields):
        encoded = field.encode("ascii")
        if len(encoded) > 32:
            parser.error(f"metadata field is too long: {field}")
        footer += encoded.ljust(32, b"\x00")
    footer += bytes(256)
    shutil.copyfile(args.library, output)
    with output.open("ab") as destination:
        destination.write(footer)
    repository = Path(__file__).resolve().parents[1]
    for filename in ("LICENSE", "THIRD_PARTY.md"):
        source = repository / filename
        destination = output.parent / filename
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    print(output)


if __name__ == "__main__":
    main()
