# TPC query SQL

TPC-H (`tpch/q1.sql`..`q22.sql`) and TPC-DS (`tpcds/01.sql`..`99.sql`), copied from
[`vortex-bench/sql` at `14a55e9e23a05a9356eee754f2ce8b69008934b6`](https://github.com/vortex-data/vortex/tree/14a55e9e23a05a9356eee754f2ce8b69008934b6/vortex-bench/sql)
so the benchmark harness can run them without depending on a Vortex checkout being present. See
[third-party sources and notices](../../THIRD_PARTY.md) for provenance and license status.

Vendored rather than referenced on purpose: a benchmark whose queries come from a path outside the repo
cannot be reproduced from the repo alone, and a silently different query set is the kind of difference
that invalidates a comparison without failing anything.

These are derived from the TPC benchmark specifications. They are not audited TPC results and must not be
described as TPC-H or TPC-DS *results* — TPC requires full disclosure and an audited kit for that. They
are the queries, used as a workload.
