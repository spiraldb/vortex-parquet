// SPDX-License-Identifier: Apache-2.0

//! Integration contracts at the DuckDB SQL boundary.

#![expect(clippy::unwrap_used)]
#![expect(clippy::tests_outside_test_module)]

use std::fs::File;
use std::fs::FileTimes;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use duckdb::Connection;
use duckdb::arrow::array::ArrayRef;
use duckdb::arrow::array::BinaryArray;
use duckdb::arrow::array::BooleanArray;
use duckdb::arrow::array::Date32Array;
use duckdb::arrow::array::Decimal128Array;
use duckdb::arrow::array::Float64Array;
use duckdb::arrow::array::Int8Array;
use duckdb::arrow::array::Int16Array;
use duckdb::arrow::array::Int32Array;
use duckdb::arrow::array::Int64Array;
use duckdb::arrow::array::ListArray;
use duckdb::arrow::array::RecordBatch;
use duckdb::arrow::array::StringArray;
use duckdb::arrow::array::StructArray;
use duckdb::arrow::array::TimestampMicrosecondArray;
use duckdb::arrow::array::UInt8Array;
use duckdb::arrow::array::UInt16Array;
use duckdb::arrow::array::UInt32Array;
use duckdb::arrow::array::UInt64Array;
use duckdb::arrow::compute::concat_batches;
use duckdb::arrow::compute::filter_record_batch;
use duckdb::arrow::datatypes::DataType;
use duckdb::arrow::datatypes::Field;
use duckdb::arrow::datatypes::Int32Type;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::EnabledStatistics;
use parquet::file::properties::WriterProperties;
use parquet::file::properties::WriterVersion;
use parquet::schema::types::ColumnPath;
use tempfile::NamedTempFile;
use vortex_parquet::ParquetFormat;
use vortex_parquet_duckdb::register_with_format;

fn fixture(batch: &RecordBatch) -> NamedTempFile {
    fixture_with_statistics(batch, EnabledStatistics::Page)
}

fn fixture_with_statistics(batch: &RecordBatch, statistics: EnabledStatistics) -> NamedTempFile {
    fixture_with_settings(batch, statistics, WriterVersion::PARQUET_1_0)
}

fn fixture_with_settings(
    batch: &RecordBatch,
    statistics: EnabledStatistics,
    version: WriterVersion,
) -> NamedTempFile {
    let file = NamedTempFile::new().unwrap();
    let properties = WriterProperties::builder()
        .set_writer_version(version)
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(3000))
        .set_data_page_row_count_limit(128)
        .set_write_batch_size(128)
        .set_statistics_enabled(statistics)
        .build();
    let mut writer =
        ArrowWriter::try_new(file.reopen().unwrap(), batch.schema(), Some(properties)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    file
}

fn numbers(rows: i64) -> RecordBatch {
    RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int64Array::from_iter_values(0..rows)) as ArrayRef,
        ),
        (
            "label",
            Arc::new(StringArray::from_iter((0..rows).map(|id| {
                (id % 7 != 0).then(|| format!("value-{id}-a-long-string-retained-by-duckdb-世界"))
            }))) as ArrayRef,
        ),
        (
            "category",
            Arc::new(StringArray::from_iter_values((0..rows).map(|id| {
                format!(
                    "group-{}-category-{}-shared-dictionary-世界",
                    id / 3000,
                    id % 5
                )
            }))) as ArrayRef,
        ),
        (
            "segment",
            Arc::new(StringArray::from_iter_values((0..rows).map(|id| {
                format!("segment-{}-{}-another-dictionary-世界", id / 3000, id % 5)
            }))) as ArrayRef,
        ),
        (
            "day",
            Arc::new(Date32Array::from_iter_values(
                (0..rows).map(|id| (id % 31) as i32),
            )) as ArrayRef,
        ),
    ])
    .unwrap()
}

struct Database(duckdb::ffi::duckdb_database);

impl Drop for Database {
    fn drop(&mut self) {
        // SAFETY: this guard owns the handle returned by duckdb_open.
        unsafe { duckdb::ffi::duckdb_close(&raw mut self.0) };
    }
}

struct SqlConnection {
    // Drop the connection before closing the database it borrows.
    connection: Connection,
    _database: Database,
}

impl Deref for SqlConnection {
    type Target = Connection;
    fn deref(&self) -> &Self::Target {
        &self.connection
    }
}

fn connection() -> (SqlConnection, ParquetFormat) {
    let mut database = Database(std::ptr::null_mut());
    // SAFETY: null path requests an in-memory database; the output belongs to our guard.
    assert_eq!(
        unsafe { duckdb::ffi::duckdb_open(std::ptr::null(), &raw mut database.0) },
        duckdb::ffi::DuckDBSuccess
    );
    let format = ParquetFormat::new();
    // SAFETY: the database guard outlives registration and the connection, in that order.
    unsafe { register_with_format(database.0, &format) }.unwrap();
    let connection = unsafe { Connection::open_from_raw(database.0) }.unwrap();
    (
        SqlConnection {
            connection,
            _database: database,
        },
        format,
    )
}

#[test]
fn integer_group_keys_preserve_bounds_across_row_groups() {
    let mut columns = Vec::new();
    let mut bounds = Vec::new();
    macro_rules! column {
        ($name:literal, $native:ty, $array:ty) => {{
            let values = (0..9000).map(|row| {
                (row % 7 != 0).then_some(match row / 3000 {
                    0 => <$native>::MIN,
                    1 => 1,
                    _ => <$native>::MAX,
                })
            });
            columns.push(($name, Arc::new(<$array>::from_iter(values)) as ArrayRef));
            bounds.push((
                $name,
                <$native>::MIN.to_string(),
                <$native>::MAX.to_string(),
            ));
        }};
    }
    column!("i8", i8, Int8Array);
    column!("i16", i16, Int16Array);
    column!("i32", i32, Int32Array);
    column!("i64", i64, Int64Array);
    column!("u8", u8, UInt8Array);
    column!("u16", u16, UInt16Array);
    column!("u32", u32, UInt32Array);
    column!("u64", u64, UInt64Array);
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    let (connection, _) = connection();
    // With statistics, each row group has a different constant bound. Without them the optimizer
    // must keep the key domain unknown. Both paths must preserve extrema and the null group.
    for statistics in [EnabledStatistics::Page, EnabledStatistics::None] {
        let file = fixture_with_statistics(&batch, statistics);
        for (name, min, max) in &bounds {
            let mut statement = connection.prepare(&format!(
                "SELECT {name}::VARCHAR, count(*) FROM read_parquet_vortex(?) GROUP BY {name} ORDER BY {name} NULLS LAST"
            )).unwrap();
            let rows = statement
                .query_map([file.path().to_str().unwrap()], |row| {
                    Ok((row.get::<_, Option<String>>(0)?, row.get::<_, i64>(1)?))
                })
                .unwrap()
                .collect::<duckdb::Result<Vec<_>>>()
                .unwrap();
            assert_eq!(
                rows,
                vec![
                    (Some(min.clone()), 2571),
                    (Some("1".to_owned()), 2571),
                    (Some(max.clone()), 2572),
                    (None, 1286),
                ],
                "{name} with {statistics:?}"
            );
            // Exact bounds may simplify this predicate to constant_or_null in DuckDB 2.0.
            // It still rejects nulls even though every non-null value satisfies the bound.
            let non_null: i64 = connection
                .query_row(
                    &format!("SELECT count(*) FROM read_parquet_vortex(?) WHERE {name} >= {min}"),
                    [file.path().to_str().unwrap()],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(non_null, 7714, "{name} with {statistics:?}");
        }
    }
}

#[test]
fn sql_streams_across_vectors_and_row_groups_and_reexecutes() {
    for (version, null_mode) in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0]
        .into_iter()
        .flat_map(|version| (0..3).map(move |mode| (version, mode)))
    {
        let base = numbers(10003);
        let mut columns = base
            .schema()
            .fields()
            .iter()
            .zip(base.columns())
            .map(|(field, array)| (field.name().clone(), Arc::clone(array), field.is_nullable()))
            .collect::<Vec<_>>();
        let narrow = (0..10003i64)
            .map(|id| {
                let valid = null_mode != 2 || id % 17 != 0;
                (
                    valid.then(|| i8::try_from(id % 256 - 128).unwrap()),
                    valid.then(|| i16::try_from((id * 97) % 65536 - 32768).unwrap()),
                    valid.then(|| u8::try_from((id * 13) % 256).unwrap()),
                    valid.then(|| u16::try_from((id * 1009) % 65536).unwrap()),
                )
            })
            .collect::<Vec<_>>();
        columns.extend([
            (
                "n8".to_owned(),
                Arc::new(Int8Array::from_iter(narrow.iter().map(|v| v.0))) as ArrayRef,
                null_mode != 0,
            ),
            (
                "n16".to_owned(),
                Arc::new(Int16Array::from_iter(narrow.iter().map(|v| v.1))) as ArrayRef,
                null_mode != 0,
            ),
            (
                "u8".to_owned(),
                Arc::new(UInt8Array::from_iter(narrow.iter().map(|v| v.2))) as ArrayRef,
                null_mode != 0,
            ),
            (
                "u16".to_owned(),
                Arc::new(UInt16Array::from_iter(narrow.iter().map(|v| v.3))) as ArrayRef,
                null_mode != 0,
            ),
        ]);
        let batch = RecordBatch::try_from_iter_with_nullable(columns).unwrap();
        let file = NamedTempFile::new().unwrap();
        // Keep string dictionaries while narrow PLAIN pages cross 2048-row output vectors.
        let mut properties = WriterProperties::builder()
            .set_writer_version(version)
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(5003))
            .set_data_page_row_count_limit(8192);
        for name in ["n8", "n16", "u8", "u16"] {
            properties = properties.set_column_dictionary_enabled(ColumnPath::from(name), false);
        }
        let mut writer = ArrowWriter::try_new(
            file.reopen().unwrap(),
            batch.schema(),
            Some(properties.build()),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let (connection, format) = connection();
        connection
            .execute_batch("SET threads=4; SET preserve_insertion_order=true")
            .unwrap();
        let mut statement = connection
            .prepare(
                "SELECT label, id, category, segment, n8, n16, u8, u16 FROM read_parquet_vortex(?) WHERE id % 3 = 0",
            )
            .unwrap();
        let expected: Vec<_> = (0..10003)
            .filter(|id| id % 3 == 0)
            .map(|id| {
                (
                    (id % 7 != 0)
                        .then(|| format!("value-{id}-a-long-string-retained-by-duckdb-世界")),
                    id,
                    format!(
                        "group-{}-category-{}-shared-dictionary-世界",
                        id / 3000,
                        id % 5
                    ),
                    format!("segment-{}-{}-another-dictionary-世界", id / 3000, id % 5),
                    narrow[usize::try_from(id).unwrap()].0,
                    narrow[usize::try_from(id).unwrap()].1,
                    narrow[usize::try_from(id).unwrap()].2,
                    narrow[usize::try_from(id).unwrap()].3,
                )
            })
            .collect();
        for _ in 0..2 {
            // Drop a partially consumed export before reusing the statement and its scan buffers.
            {
                let mut partial = statement.query([file.path().to_str().unwrap()]).unwrap();
                assert_eq!(
                    partial.next().unwrap().unwrap().get::<_, i64>(1).unwrap(),
                    0
                );
            }
            let rows = statement
                .query_map([file.path().to_str().unwrap()], |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<i8>>(4)?,
                        row.get::<_, Option<i16>>(5)?,
                        row.get::<_, Option<u8>>(6)?,
                        row.get::<_, Option<u16>>(7)?,
                    ))
                })
                .unwrap()
                .collect::<duckdb::Result<Vec<_>>>()
                .unwrap();
            assert_eq!(rows.len(), expected.len());
            for (index, (row, expected)) in rows.iter().zip(&expected).enumerate() {
                assert_eq!(row, expected, "row {index}");
            }
        }
        assert!(format.metrics().snapshot().page_decodes > 0);
        let joined: i64 = connection.query_row(
            "SELECT count(*) FROM read_parquet_vortex(?1) a JOIN read_parquet_vortex(?1) b USING (id)",
            [file.path().to_str().unwrap()], |row| row.get(0),
        ).unwrap();
        assert_eq!(joined, 10003);
        // DuckDB retains the build-side strings while the probe scan advances and drops its buffers.
        let matched: i64 = connection.query_row(
            "SELECT count(*) FROM read_parquet_vortex(?1) a JOIN read_parquet_vortex(?1) b ON a.label = b.label AND a.category = b.category AND a.n16 IS NOT DISTINCT FROM b.n16",
            [file.path().to_str().unwrap()], |row| row.get(0),
        ).unwrap();
        assert_eq!(matched, 10003 - (10003 + 6) / 7);
    }
}

#[test]
fn filtered_joins_preserve_results_across_reexecution_and_early_stop() {
    let count = 40003i64;
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int64Array::from_iter_values(0..count)) as ArrayRef,
        ),
        (
            "compact_key",
            Arc::new(Int64Array::from_iter(
                (0..count).map(|id| (id % 17 != 0).then_some(id - 20000)),
            )) as ArrayRef,
        ),
        (
            "key",
            Arc::new(Int64Array::from_iter(
                (0..count).map(|id| (id % 17 != 0).then_some((id - 20000) * 100003)),
            )) as ArrayRef,
        ),
    ])
    .unwrap();
    let file = fixture(&batch);
    let (connection, _) = connection();
    let path = file.path().to_str().unwrap();
    // Dense and sparse key domains exercise DuckDB's prefix-range and Bloom join hints.
    // Both must preserve null semantics and replace their state on every execution.
    for key in ["compact_key", "key"] {
        for comparison in ["=", "IS NOT DISTINCT FROM"] {
            for threads in [1, 4] {
                connection
                    .execute_batch(&format!(
                        "SET threads={threads}; SET perfect_ht_threshold=0"
                    ))
                    .unwrap();
                let mut joined = connection
                    .prepare(&format!(
                        "SELECT count(*), sum(a.id)::BIGINT FROM read_parquet_vortex(?1) a
                     JOIN read_parquet_vortex(?1) b ON a.{key} {comparison} b.{key} WHERE b.id % 97 = ?2"
                    ))
                    .unwrap();
                for residue in [3i64, 41, 3] {
                    // Drop an incompletely consumed join, then execute a new filter with the same files.
                    connection
                        .query_row(
                            &format!("SELECT a.id FROM read_parquet_vortex(?1) a JOIN read_parquet_vortex(?1) b
                         ON a.{key} {comparison} b.{key} WHERE b.id % 97 = ?2 LIMIT 1"),
                            duckdb::params![path, residue],
                            |row| row.get::<_, i64>(0),
                        )
                        .unwrap();
                    let actual: (i64, i64) = joined
                        .query_row(duckdb::params![path, residue], |row| {
                            Ok((row.get(0)?, row.get(1)?))
                        })
                        .unwrap();
                    let ids: Vec<_> = (0..count)
                        .filter(|id| id % 97 == residue && id % 17 != 0)
                        .collect();
                    let null_matches = if comparison == "=" {
                        0
                    } else {
                        (0..count)
                            .filter(|id| id % 97 == residue && id % 17 == 0)
                            .count() as i64
                    };
                    let null_ids: Vec<_> = (0..count).filter(|id| id % 17 == 0).collect();
                    assert_eq!(
                        actual,
                        (
                            ids.len() as i64 + null_matches * null_ids.len() as i64,
                            ids.iter().sum::<i64>() + null_matches * null_ids.iter().sum::<i64>()
                        )
                    );
                }
            }
        }
    }
}

#[test]
fn large_membership_scans_preserve_residuals_and_distinct_subplans() {
    let count = (1 << 20) + 3i64;
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int64Array::from_iter_values(0..count)) as ArrayRef,
        ),
        (
            "k",
            Arc::new(Int64Array::from_iter(
                (0..count).map(|i| (i % 17 != 0).then_some(i % 9)),
            )) as ArrayRef,
        ),
        (
            "f",
            Arc::new(Float64Array::from_iter_values(
                (0..count).map(|i| if i % 5 == 0 { f64::NAN } else { i as f64 }),
            )) as ArrayRef,
        ),
        (
            "st",
            Arc::new(StructArray::from(vec![(
                Arc::new(Field::new("x", DataType::Int64, false)),
                Arc::new(Int64Array::from_iter_values((0..count).map(|i| i % 9))) as ArrayRef,
            )])) as ArrayRef,
        ),
    ])
    .unwrap();
    let file = NamedTempFile::new().unwrap();
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(131072))
        .build();
    let mut writer =
        ArrowWriter::try_new(file.reopen().unwrap(), batch.schema(), Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let (connection, _) = connection();
    let path = file.path().to_str().unwrap();
    let plan: String = connection.query_row(
        "EXPLAIN (FORMAT JSON) SELECT count(*) FROM read_parquet_vortex(?) WHERE k IN (1, 4, 7)",
        [path], |row| row.get(1),
    ).unwrap();
    assert!(
        !plan.contains("\"name\": \"FILTER\""),
        "membership must be fully evaluated by the large scan: {plan}"
    );
    type Predicate = fn(i64) -> bool;
    for (predicate, accepts) in [
        (
            "k IN (1, 4, 7)",
            (|i| i % 17 != 0 && [1, 4, 7].contains(&(i % 9))) as Predicate,
        ),
        (
            "k < 2 OR k > 7",
            (|i| i % 17 != 0 && (i % 9 < 2 || i % 9 > 7)) as Predicate,
        ),
        (
            "1 > k OR k = 8",
            (|i| i % 17 != 0 && [0, 8].contains(&(i % 9))) as Predicate,
        ),
        (
            "k IN (1, 4) AND id >= 1000 AND id % 13 = 0",
            (|i| i % 17 != 0 && [1, 4].contains(&(i % 9)) && i >= 1000 && i % 13 == 0) as Predicate,
        ),
        (
            "k IN (1, 4) AND f < 100",
            (|i| i % 17 != 0 && [1, 4].contains(&(i % 9)) && i % 5 != 0 && i < 100) as Predicate,
        ),
        (
            "k IN (1, 4) AND st IN ({x: 1}, {x: 4})",
            (|i| i % 17 != 0 && [1, 4].contains(&(i % 9))) as Predicate,
        ),
        (
            "k IN (1, NULL)",
            (|i| i % 17 != 0 && i % 9 == 1) as Predicate,
        ),
        ("k NOT IN (1, NULL)", (|_| false) as Predicate),
        (
            "k IN (1, 4) AND id > 10 AND id < 5",
            (|_| false) as Predicate,
        ),
    ] {
        let expected = (0..count)
            .filter(|&i| accepts(i))
            .fold((0i64, 0i64), |(n, sum), i| (n + 1, sum + i));
        let actual: (i64, i64) = connection.query_row(
            &format!("SELECT count(*), coalesce(sum(id), 0)::BIGINT FROM read_parquet_vortex(?) WHERE {predicate}"),
            [path], |row| Ok((row.get(0)?, row.get(1)?)),
        ).unwrap();
        assert_eq!(actual, expected, "{predicate}");
    }
    let mut statement = connection
        .prepare(
            "SELECT (SELECT count(*) FROM read_parquet_vortex(?1) WHERE k IN (?2, ?3)),
                (SELECT count(*) FROM read_parquet_vortex(?1) WHERE k IN (2, 5, 8))",
        )
        .unwrap();
    for (a, b) in [(1i64, 4i64), (0, 7), (1, 4)] {
        let actual: (i64, i64) = statement
            .query_row(duckdb::params![path, a, b], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        let expected = (0..count)
            .filter(|i| i % 17 != 0)
            .fold((0i64, 0i64), |(left, right), i| {
                (
                    left + i64::from([a, b].contains(&(i % 9))),
                    right + i64::from([2, 5, 8].contains(&(i % 9))),
                )
            });
        assert_eq!(actual, expected);
    }
}

#[test]
fn file_lists_preserve_global_bounds_duplicates_and_reexecution() {
    let make_file = |ids: Vec<i64>| {
        fixture(
            &RecordBatch::try_from_iter([("id", Arc::new(Int64Array::from(ids)) as ArrayRef)])
                .unwrap(),
        )
    };
    let first = make_file(vec![1, 2, 2]);
    let second = make_file(vec![1 << 40, (1 << 40) + 1]);
    let empty = make_file(vec![]);
    let (connection, _) = connection();
    for threads in [1, 4] {
        connection
            .execute_batch(&format!("SET threads={threads}"))
            .unwrap();
        let mut statement = connection
            .prepare(
                "SELECT id, count(*) FROM read_parquet_vortex([?1, ?2, ?1])
             WHERE id >= ?3 GROUP BY id ORDER BY id",
            )
            .unwrap();
        for cutoff in [0i64, 1 << 40, 0] {
            let rows = statement
                .query_map(
                    duckdb::params![
                        first.path().to_str().unwrap(),
                        second.path().to_str().unwrap(),
                        cutoff
                    ],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                )
                .unwrap()
                .collect::<duckdb::Result<Vec<_>>>()
                .unwrap();
            let expected: Vec<_> = [(1, 2), (2, 4), (1 << 40, 1), ((1 << 40) + 1, 1)]
                .into_iter()
                .filter(|(id, _)| *id >= cutoff)
                .collect();
            assert_eq!(rows, expected);
        }
        let count: i64 = connection
            .query_row(
                "SELECT count(*) FROM read_parquet_vortex([?1, ?2, ?3, ?2])",
                duckdb::params![
                    empty.path().to_str().unwrap(),
                    first.path().to_str().unwrap(),
                    second.path().to_str().unwrap()
                ],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 8);
        // File-list order, duplicate files, and row-group order survive parallel result collection.
        let rows = connection
            .prepare("SELECT id FROM read_parquet_vortex([?1, ?2, ?1])")
            .unwrap()
            .query_map(
                duckdb::params![
                    first.path().to_str().unwrap(),
                    second.path().to_str().unwrap()
                ],
                |row| row.get::<_, i64>(0),
            )
            .unwrap()
            .collect::<duckdb::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(rows, [1, 2, 2, 1 << 40, (1 << 40) + 1, 1, 2, 2]);
        // Empty inputs must not shrink the optimizer's bounds or prevent reaching later files.
        let high: i64 = connection
            .query_row(
                "SELECT max(id) FROM read_parquet_vortex([?1, ?2])",
                duckdb::params![
                    empty.path().to_str().unwrap(),
                    second.path().to_str().unwrap()
                ],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(high, (1 << 40) + 1);
    }
    for argument in ["[]", "[NULL]", "[1]", "42"] {
        assert!(
            connection
                .prepare(&format!("SELECT * FROM read_parquet_vortex({argument})"))
                .is_err()
        );
    }
    let wrong_schema = fixture(&numbers(1));
    assert!(
        connection
            .query_row(
                "SELECT count(*) FROM read_parquet_vortex([?1, ?2])",
                duckdb::params![
                    first.path().to_str().unwrap(),
                    wrong_schema.path().to_str().unwrap()
                ],
                |row| row.get::<_, i64>(0),
            )
            .is_err()
    );
    // Failed list binds must not poison the connection or the metadata cache.
    let count: i64 = connection
        .query_row(
            "SELECT count(*) FROM read_parquet_vortex(?)",
            [first.path().to_str().unwrap()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 3);
}

#[test]
fn projection_count_and_early_stop_avoid_unneeded_decoding() {
    let file = fixture(&numbers(10003));
    let (connection, format) = connection();
    // A one-worker scan must stop without decoding later row groups. Parallel scans may already
    // have one row group in flight per worker when DuckDB satisfies LIMIT.
    connection.execute_batch("SET threads=1").unwrap();
    let path = file.path().to_str().unwrap();
    let count: i64 = connection
        .query_row(
            "SELECT count(*) FROM read_parquet_vortex(?)",
            [path],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 10003);
    assert_eq!(format.metrics().snapshot().page_decodes, 0);
    format.metrics().reset();

    let mut statement = connection
        .prepare(&format!(
            "SELECT id FROM read_parquet_vortex('{}') LIMIT 1",
            path.replace('\'', "''")
        ))
        .unwrap();
    assert_eq!(format.metrics().snapshot().page_decodes, 0);
    let first: i64 = statement.query_row([], |row| row.get(0)).unwrap();
    assert_eq!(first, 0);
    assert_eq!(format.metrics().snapshot().page_decodes, 1);
    drop(statement);

    format.metrics().reset();
    let sum: i64 = connection
        .query_row(
            "SELECT sum(id)::BIGINT FROM read_parquet_vortex(?)",
            [path],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(sum, 10002 * 10003 / 2);
    // Four row groups, one selected column. The label column must remain unread.
    assert_eq!(format.metrics().snapshot().page_decodes, 4);
    // A full-column scan cannot skip a page, so it must not fetch selection indexes.
    assert_eq!(format.metrics().snapshot().offset_index_decodes, 0);

    // A filter disproved by every row-group footer must not read any column payload.
    format.metrics().reset();
    let absent: i64 = connection
        .query_row(
            "SELECT count(*) FROM read_parquet_vortex(?) WHERE id = -1",
            [path],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(absent, 0);
    assert_eq!(format.metrics().snapshot().page_decodes, 0);

    // A string predicate rejecting every row must not decode the projected id column.
    format.metrics().reset();
    let absent_sum: Option<i64> = connection
        .query_row(
            "SELECT sum(id)::BIGINT FROM read_parquet_vortex(?) WHERE label LIKE '%not-a-match%'",
            [path],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(absent_sum, None);
    assert_eq!(format.metrics().snapshot().page_decodes, 4);

    // Top-N tightens its bound after the first row group. Later groups must be pruned, and
    // re-executing the prepared query must start with a fresh bound rather than the previous top-N.
    let mut top = connection
        .prepare(
            "SELECT id, label FROM read_parquet_vortex(?1) WHERE id >= ?2 ORDER BY id LIMIT 10",
        )
        .unwrap();
    for start in [0i64, 5000] {
        format.metrics().reset();
        let rows = top
            .query_map(duckdb::params![path, start], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
            })
            .unwrap()
            .collect::<duckdb::Result<Vec<_>>>()
            .unwrap();
        let expected: Vec<_> = (start..start + 10)
            .map(|id| {
                (
                    id,
                    (id % 7 != 0)
                        .then(|| format!("value-{id}-a-long-string-retained-by-duckdb-世界")),
                )
            })
            .collect();
        assert_eq!(rows, expected);
        let decodes = format.metrics().snapshot().page_decodes;
        assert!(decodes < 8, "{decodes}");
    }

    // Parameter rebinding and reordered projection must preserve original column identities.
    let mut filtered = connection
        .prepare(
            "SELECT label, epoch(day)::BIGINT, id FROM read_parquet_vortex(?1) WHERE id >= ?2 AND id < ?3 AND id <> ?4 AND day < DATE '1970-01-20' ORDER BY id",
        )
        .unwrap();
    for start in [0i64, 3000] {
        let rows = filtered
            .query_map(
                duckdb::params![path, start, start + 2000, start + 1],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .unwrap()
            .collect::<duckdb::Result<Vec<_>>>()
            .unwrap();
        let expected: Vec<_> = (start..start + 2000)
            .filter(|id| *id != start + 1 && id % 31 < 19)
            .map(|id| {
                (
                    (id % 7 != 0)
                        .then(|| format!("value-{id}-a-long-string-retained-by-duckdb-世界")),
                    (id % 31) * 86400,
                    id,
                )
            })
            .collect();
        assert_eq!(rows, expected);
    }
}

#[test]
fn logical_and_nested_values_keep_their_sql_meaning() {
    let list = ListArray::from_iter_primitive::<Int32Type, _, _>([
        Some(vec![Some(1), None, Some(3)]),
        None,
        Some(vec![]),
    ]);
    let child = Arc::new(StringArray::from(vec![Some("nested"), None, Some("last")])) as ArrayRef;
    let structure = StructArray::from(vec![(
        Arc::new(Field::new("name", child.data_type().clone(), true)),
        child,
    )]);
    let batch = RecordBatch::try_from_iter([
        ("id", Arc::new(Int32Array::from(vec![0, 1, 2])) as ArrayRef),
        ("items", Arc::new(list) as ArrayRef),
        ("detail", Arc::new(structure) as ArrayRef),
        (
            "amount",
            Arc::new(
                Decimal128Array::from(vec![Some(12345), None, Some(-100)])
                    .with_precision_and_scale(12, 2)
                    .unwrap(),
            ) as ArrayRef,
        ),
        (
            "day",
            Arc::new(Date32Array::from(vec![Some(0), None, Some(1)])) as ArrayRef,
        ),
        (
            "ts",
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1234567),
                None,
                Some(-1),
            ])) as ArrayRef,
        ),
        (
            "blob",
            Arc::new(BinaryArray::from(vec![
                Some(&b"a\0b"[..]),
                None,
                Some(&b""[..]),
            ])) as ArrayRef,
        ),
        (
            "measure",
            Arc::new(Float64Array::from(vec![Some(f64::NAN), None, Some(-0.0)])) as ArrayRef,
        ),
        (
            "text",
            Arc::new(StringArray::from(vec![
                Some("%_\\世界\n\0end"),
                None,
                Some(""),
            ])) as ArrayRef,
        ),
    ])
    .unwrap();
    let file = fixture(&batch);
    let (connection, _) = connection();
    let mismatches: i64 = connection
        .query_row(
            "WITH actual AS (SELECT * FROM read_parquet_vortex(?)), expected AS (
            SELECT 0 id, [1,NULL,3]::INTEGER[] items, {'name':'nested'} detail,
                   123.45::DECIMAL(12,2) amount, DATE '1970-01-01' AS day_value,
                   TIMESTAMP '1970-01-01 00:00:01.234567' ts, 'a\\x00b'::BLOB blob, 'NaN'::DOUBLE measure,
                   '%_\\世界' || chr(10) || chr(0) || 'end' AS text
            UNION ALL SELECT 1, NULL, {'name':NULL}, NULL, NULL, NULL, NULL, NULL, NULL
            UNION ALL SELECT 2, [], {'name':'last'}, -1.00, DATE '1970-01-02',
                   TIMESTAMP '1969-12-31 23:59:59.999999', ''::BLOB, -0.0::DOUBLE, ''
        ) SELECT count(*) FROM (
            (SELECT * FROM actual EXCEPT ALL SELECT * FROM expected)
            UNION ALL (SELECT * FROM expected EXCEPT ALL SELECT * FROM actual)
        )",
            [file.path().to_str().unwrap()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(mismatches, 0);

    for (predicate, expected) in [
        ("amount >= -1.00 AND amount < 124.00", vec![0, 2]),
        ("amount IS NULL OR amount < 0", vec![1, 2]),
        (
            "day >= DATE '1970-01-02' AND ts < TIMESTAMP '1970-01-01'",
            vec![2],
        ),
        ("detail.name IN ('nested', 'last')", vec![0, 2]),
        ("detail.name IN ('nested', NULL)", vec![0]),
        ("text IN ('', NULL)", vec![2]),
        ("text NOT IN ('', NULL)", vec![]),
        ("id IN (0, NULL)", vec![0]),
        ("amount IN (123.45, NULL)", vec![0]),
        ("day IN (DATE '1970-01-01', NULL)", vec![0]),
        ("detail.name IS NULL", vec![1]),
        ("blob = 'a\\x00b'::BLOB", vec![0]),
        ("amount IS NOT NULL AND id IN (0, 2)", vec![0, 2]),
        ("items IS NULL", vec![1]),
        ("measure = 'NaN'::DOUBLE", vec![0]),
        ("measure > 0", vec![0]),
        ("measure = 0", vec![2]),
        ("contains(text, '%_\\')", vec![0]),
        ("contains(text, '世界')", vec![0]),
        ("contains(text, chr(0))", vec![0]),
        ("prefix(text, '%_')", vec![0]),
        ("suffix(text, 'end')", vec![0]),
        ("text LIKE '%世界%'", vec![0]),
        ("contains(text, '')", vec![0, 2]),
        ("prefix(text, '')", vec![0, 2]),
        ("suffix(text, '')", vec![0, 2]),
        ("contains(text, NULL)", vec![]),
        ("contains(text, 'missing') OR text IS NULL", vec![1]),
        ("text ILIKE '%END%'", vec![0]),
    ] {
        let sql = format!("SELECT id FROM read_parquet_vortex(?) WHERE {predicate} ORDER BY id");
        let actual = connection
            .prepare(&sql)
            .unwrap()
            .query_map([file.path().to_str().unwrap()], |row| row.get::<_, i32>(0))
            .unwrap()
            .collect::<duckdb::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(actual, expected, "{predicate}");
    }
}

#[test]
fn empty_files_and_errors_leave_the_connection_usable() {
    let empty = fixture(&numbers(0));
    let corrupt = NamedTempFile::new().unwrap();
    let (connection, format) = connection();
    // Re-registration through a temporary connection must retain the catalog callback state.
    // SAFETY: the fixture owns a live database for this call and the subsequent queries.
    unsafe { register_with_format(connection._database.0, &format) }.unwrap();
    for path in [corrupt.path().to_str().unwrap(), "/does/not/exist.parquet"] {
        assert!(
            connection
                .prepare(&format!("SELECT * FROM read_parquet_vortex('{path}')"))
                .is_err()
        );
    }
    assert!(
        connection
            .prepare("SELECT * FROM read_parquet_vortex(NULL)")
            .is_err()
    );
    // Bind a valid footer, then make the lazy payload read fail during execution.
    let truncated = fixture(&numbers(10003));
    let mut statement = connection
        .prepare(&format!(
            "SELECT sum(id) FROM read_parquet_vortex('{}')",
            truncated.path().to_str().unwrap().replace('\'', "''")
        ))
        .unwrap();
    truncated.as_file().set_len(8).unwrap();
    assert!(statement.query_row([], |row| row.get::<_, i64>(0)).is_err());
    drop(statement);
    // A new bind must observe file changes even when size and modification time are unchanged.
    // The cache must retain only metadata: every full scan still decodes its column data.
    let changing = NamedTempFile::new().unwrap();
    let write_values = |path: &Path, name: &str, values: Vec<i64>| {
        let batch =
            RecordBatch::try_from_iter([(name, Arc::new(Int64Array::from(values)) as ArrayRef)])
                .unwrap();
        let properties = WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(path).unwrap(),
            batch.schema(),
            Some(properties),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    };
    write_values(changing.path(), "id", vec![1, 1, 1]);
    let attributes = changing.as_file().metadata().unwrap();
    let changed_path = changing.path().to_str().unwrap();
    for _ in 0..2 {
        format.metrics().reset();
        let sum: i64 = connection
            .query_row(
                "SELECT sum(id)::BIGINT FROM read_parquet_vortex(?)",
                [changed_path],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(sum, 3);
        assert_eq!(format.metrics().snapshot().page_decodes, 1);
    }
    write_values(changing.path(), "id", vec![9, 9, 9]);
    changing
        .as_file()
        .set_times(FileTimes::new().set_modified(attributes.modified().unwrap()))
        .unwrap();
    assert_eq!(
        changing.as_file().metadata().unwrap().len(),
        attributes.len()
    );
    let count: i64 = connection
        .query_row(
            "SELECT count(*) FROM read_parquet_vortex(?) WHERE id = 9",
            [changed_path],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 3);

    // Replacing the path also changes schema and cardinality. Missing/corrupt replacements must
    // error rather than reusing a cached footer from a formerly valid file at that path.
    let replacement = NamedTempFile::new().unwrap();
    write_values(replacement.path(), "fresh", vec![5, 6]);
    std::fs::rename(replacement.path(), changing.path()).unwrap();
    let result: (i64, i64) = connection
        .query_row(
            "SELECT count(*), sum(fresh)::BIGINT FROM read_parquet_vortex(?)",
            [changed_path],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(result, (2, 11));
    std::fs::remove_file(changing.path()).unwrap();
    assert!(
        connection
            .query_row(
                "SELECT count(*) FROM read_parquet_vortex(?)",
                [changed_path],
                |row| row.get::<_, i64>(0)
            )
            .is_err()
    );
    std::fs::write(changing.path(), b"corrupt").unwrap();
    assert!(
        connection
            .query_row(
                "SELECT count(*) FROM read_parquet_vortex(?)",
                [changed_path],
                |row| row.get::<_, i64>(0)
            )
            .is_err()
    );

    let count: i64 = connection
        .query_row(
            "SELECT count(*) FROM read_parquet_vortex(?)",
            [empty.path().to_str().unwrap()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 0);
    let mut statement = connection
        .prepare("SELECT * FROM read_parquet_vortex(?)")
        .unwrap();
    assert!(
        statement
            .query([empty.path().to_str().unwrap()])
            .unwrap()
            .next()
            .unwrap()
            .is_none()
    );
}

#[test]
fn decimal_storage_boundaries_and_invalid_precision_are_preserved() {
    let cases = [
        ("small", 4, 9_999i128),
        ("medium", 9, 999_999_999),
        ("large", 18, 999_999_999_999_999_999),
        (
            "huge",
            38,
            99_999_999_999_999_999_999_999_999_999_999_999_999,
        ),
    ];
    let (connection, _) = connection();
    // Distinct values exceed the small-dictionary threshold, span data pages and row groups,
    // and cross DuckDB vector boundaries. Both nullable and non-null codes must preserve values.
    for nullable in [false, true] {
        let mut columns = cases
            .map(|(name, precision, bound)| {
                (
                    name,
                    Arc::new(
                        Decimal128Array::from_iter((0..4101i128).map(|index| {
                            if nullable && index % 11 == 2 {
                                None
                            } else {
                                let value = bound - index / 2;
                                Some(if index % 2 == 0 { value } else { -value })
                            }
                        }))
                        .with_precision_and_scale(precision, 2)
                        .unwrap(),
                    ) as ArrayRef,
                )
            })
            .to_vec();
        columns.push((
            "id",
            Arc::new(Int32Array::from_iter_values(0..4101)) as ArrayRef,
        ));
        let batch = RecordBatch::try_from_iter(columns).unwrap();
        let file = fixture(&batch);
        for filtered in [false, true] {
            let predicate = if filtered { " WHERE small >= 0" } else { "" };
            let sql = format!("SELECT * FROM read_parquet_vortex(?){predicate} ORDER BY id");
            let mut statement = connection.prepare(&sql).unwrap();
            let actual: Vec<_> = statement
                .query_arrow([file.path().to_str().unwrap()])
                .unwrap()
                .collect();
            let expected = if filtered {
                let keep = BooleanArray::from(
                    (0..4101)
                        .map(|index| index % 2 == 0 && !(nullable && index % 11 == 2))
                        .collect::<Vec<_>>(),
                );
                filter_record_batch(&batch, &keep).unwrap()
            } else {
                batch.clone()
            };
            assert_eq!(
                concat_batches(&batch.schema(), &actual).unwrap(),
                expected,
                "nullable={nullable}, filtered={filtered}"
            );
        }
    }

    // Precision must be checked both when storage is narrowed and when it is exported directly.
    for (precision, sign) in [4, 9, 18, 38]
        .into_iter()
        .flat_map(|precision| [-1, 1].map(|sign| (precision, sign)))
    {
        let invalid = RecordBatch::try_from_iter([(
            "amount",
            Arc::new(
                Decimal128Array::from(vec![sign * 10i128.pow(u32::from(precision))])
                    .with_precision_and_scale(precision, 0)
                    .unwrap(),
            ) as ArrayRef,
        )])
        .unwrap();
        let file = fixture(&invalid);
        let error = connection
            .query_row(
                "SELECT amount FROM read_parquet_vortex(?)",
                [file.path().to_str().unwrap()],
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("precision"), "{error}");
    }
}
