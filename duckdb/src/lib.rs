// SPDX-License-Identifier: Apache-2.0

//! DuckDB table function for scanning Parquet with the Vortex reader.
//!
//! Call [`register`] once, then query `read_parquet_vortex('path.parquet')`.
//! Files are opened at bind time; each execution has a fresh, lazy scan. Projection and supported
//! native scan filters are pushed into Vortex. Other expressions remain in DuckDB.
//! DuckDB workers claim independent row groups and each holds at most one
//! Vortex batch at a time, exported directly into native DuckDB vectors.

#[cfg(all(feature = "bundled", feature = "loadable-extension"))]
compile_error!(
    "bundled and loadable-extension are alternative DuckDB linking modes; build them separately"
);

mod bloom;
mod export;
mod filter;
mod local_io;
mod metadata_cache;
mod registration;
mod snappy;
mod statistics;

use std::error::Error;
use std::ffi::CStr;
use std::sync::Arc;

use duckdb::core::LogicalTypeId;
use duckdb::vtab::BindInfo;
use duckdb::vtab::InitInfo;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::expr::root;
use vortex::expr::select;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::DataSourceRef;
use vortex::scan::PartitionStream;
use vortex::scan::ScanRequest;
use vortex::session::VortexSession;
use vortex_parquet::ParquetFormat;
use vortex_parquet::PreparedParquetMetadata;
use vortex_parquet::file::FileHandle;

use crate::export::ArrayExporter;
use crate::export::logical_type;
use crate::local_io::WorkerFileReadAt;
use crate::metadata_cache::CachedMetadata;
use crate::metadata_cache::FileVersion;
use crate::metadata_cache::MetadataCache;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Registers `read_parquet_vortex(path)` in an upstream DuckDB database.
///
/// # Safety
/// `database` must be a live DuckDB 2.0.0 database handle. It is borrowed for this call;
/// DuckDB owns the registered callbacks and releases their state with the catalog.
pub unsafe fn register(database: duckdb::ffi::duckdb_database) -> duckdb::Result<()> {
    // SAFETY: forwarded from the caller's database lifetime contract.
    unsafe { register_with_format(database, &ParquetFormat::new()) }
}

/// Registers the table function with a shared reader configuration, cache, and metrics.
///
/// # Safety
/// `database` must satisfy the lifetime and version contract of [`register`].
pub unsafe fn register_with_format(
    database: duckdb::ffi::duckdb_database,
    format: &ParquetFormat,
) -> duckdb::Result<()> {
    // The ownership shim accesses DuckDB's C++ vector layout, which is version-specific.
    // SAFETY: DuckDB returns a static, NUL-terminated library version string.
    let version = unsafe { CStr::from_ptr(duckdb::ffi::duckdb_library_version()) };
    check_version(version)?;
    let runtime = CurrentThreadRuntime::new();
    let context = ReaderContext {
        session: VortexSession::default().with_handle(runtime.handle()),
        runtime,
        format: format
            .clone()
            .with_options((*format.options()).with_snappy_append_decoder(snappy::decode)),
        metadata: Arc::default(),
    };
    // SAFETY: the native bridge borrows the database and takes ownership of this context.
    unsafe { registration::register(database, context) }
}

fn check_version(version: &CStr) -> duckdb::Result<()> {
    if version.to_bytes() != b"v2.0.0" {
        return Err(duckdb::Error::DuckDBFailure(
            duckdb::ffi::Error::new(duckdb::ffi::DuckDBError),
            Some(format!(
                "vortex-parquet requires DuckDB v2.0.0 for its native vector ABI; found {}",
                version.to_string_lossy()
            )),
        ));
    }
    Ok(())
}

struct ParquetVTab;

/// Reuse registries and the runtime across file binds. File readers and query cursors
/// remain separate; cloning the runtime lets DuckDB workers drive their own futures concurrently.
#[derive(Clone)]
struct ReaderContext {
    runtime: CurrentThreadRuntime,
    session: VortexSession,
    format: ParquetFormat,
    metadata: Arc<Mutex<MetadataCache>>,
}

struct BindData {
    runtime: CurrentThreadRuntime,
    session: VortexSession,
    source: DataSourceRef,
    additional_sources: Vec<DataSourceRef>,
    metadata: Vec<Arc<CachedMetadata>>,
}

struct InitData {
    blooms: bloom::Bindings,
    partitions: Mutex<Partitions>,
    vector_size: usize,
}

struct Partitions {
    stream: PartitionStream,
    next_index: u64,
}

#[derive(Default)]
struct Scan {
    partition_index: u64,
    stream: Option<BoxStream<'static, VortexResult<ArrayRef>>>,
    pending: Option<ArrayExporter>,
}

impl ParquetVTab {
    fn bind(bind: &BindInfo) -> Result<BindData> {
        let path = bind.get_parameter(0);
        if path.is_null() {
            return Err("read_parquet_vortex requires a non-NULL local path".into());
        }
        let paths = path.to_list().unwrap_or_else(|| vec![path]);
        if paths.is_empty() {
            return Err("read_parquet_vortex requires at least one local path".into());
        }
        let paths = paths
            .into_iter()
            .map(|path| {
                if path.is_null() || path.logical_type_id() != LogicalTypeId::Varchar {
                    return Err("read_parquet_vortex paths must be non-NULL strings".into());
                }
                Ok(path.to_string())
            })
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: register_with_format installs a ReaderContext as this function's extra info;
        // DuckDB owns it for the lifetime of the registered function.
        let context = unsafe { &*bind.get_extra_info::<ReaderContext>() };
        let runtime = context.runtime.clone();
        let session = context.session.clone();
        let (mut sources, metadata) = runtime.block_on(async {
            let mut sources = Vec::with_capacity(paths.len());
            let mut cached_metadata = Vec::with_capacity(paths.len());
            for path in paths {
                let read = Arc::new(WorkerFileReadAt::open(path, session.handle())?);
                let attributes = read.metadata()?;
                let version = FileVersion::from_metadata(&attributes);
                let file = FileHandle::new(
                    Arc::<WorkerFileReadAt>::clone(&read),
                    attributes.len(),
                    session.clone(),
                );
                let cached = version.and_then(|version| context.metadata.lock().get(version));
                let metadata = if let Some(metadata) = cached {
                    metadata
                } else {
                    let metadata = Arc::new(CachedMetadata::new(PreparedParquetMetadata::new(
                        &file,
                        vortex_parquet::file_metadata(&file).await?,
                    )?));
                    // Do not retain a footer if the file changed while it was being read.
                    if let Some(version) = version
                        && Some(version) == FileVersion::from_metadata(&read.metadata()?)
                    {
                        context
                            .metadata
                            .lock()
                            .insert(version, Arc::clone(&metadata));
                    }
                    metadata
                };
                sources.push(
                    context
                        .format
                        .open_data_source_with_prepared_metadata(file, &metadata.prepared)?,
                );
                cached_metadata.push(metadata);
            }
            Ok::<_, vortex::error::VortexError>((sources, cached_metadata))
        })?;
        let source = sources.remove(0);
        if sources.iter().any(|other| other.dtype() != source.dtype()) {
            return Err("read_parquet_vortex requires matching schemas across files".into());
        }
        let fields = source
            .dtype()
            .as_struct_fields_opt()
            .ok_or("Parquet root must be a struct")?;
        for (name, dtype) in fields.names().iter().zip(fields.fields()) {
            bind.add_result_column(name.as_ref(), logical_type(&dtype)?);
        }
        let rows = std::iter::once(&source)
            .chain(&sources)
            .try_fold(0u64, |sum, source| {
                sum.checked_add(source.row_count().as_exact()?)
            });
        if let Some(rows) = rows {
            bind.set_cardinality(rows, true);
        }
        Ok(BindData {
            runtime,
            session,
            source,
            additional_sources: sources,
            metadata,
        })
    }

    fn init(init: &InitInfo) -> Result<InitData> {
        // SAFETY: DuckDB supplies the BindData returned by this VTab's bind callback.
        let bind = unsafe { &*init.get_bind_data::<BindData>() };
        let columns = filter::projection();
        let names: FieldNames = columns
            .iter()
            .map(|&index| {
                let index = usize::try_from(index)?;
                let fields = bind
                    .source
                    .dtype()
                    .as_struct_fields_opt()
                    .ok_or("Parquet root must be a struct")?;
                let field = fields
                    .names()
                    .get(index)
                    .ok_or("invalid projected column")?;
                Ok(FieldName::from(field.as_ref()))
            })
            .collect::<Result<_>>()?;
        let projection = select(names, root());
        let filter = filter::current(bind.source.dtype())?;
        let mut streams = Vec::with_capacity(1 + bind.additional_sources.len());
        let mut partitions = 0usize;
        for source in std::iter::once(&bind.source).chain(&bind.additional_sources) {
            let scan = bind.runtime.block_on(source.scan(ScanRequest {
                projection: projection.clone(),
                filter: filter.clone(),
                ordered: true,
                ..ScanRequest::default()
            }))?;
            partitions = partitions
                .checked_add(scan.partition_count().as_exact().unwrap_or(1))
                .ok_or("scan partition count overflow")?;
            streams.push(scan.partitions());
        }
        init.set_max_threads(u64::try_from(partitions.max(1))?);
        let partitions = if streams.len() == 1 {
            streams.pop().ok_or("missing scan partition stream")?
        } else {
            futures::stream::iter(streams).flatten().boxed()
        };
        // SAFETY: DuckDB is initialized before invoking table-function callbacks.
        let vector_size = usize::try_from(unsafe { duckdb::ffi::duckdb_vector_size() })?;
        if vector_size == 0 {
            return Err("DuckDB returned a zero vector size".into());
        }
        Ok(InitData {
            blooms: bloom::Bindings::capture(),
            partitions: Mutex::new(Partitions {
                stream: partitions,
                next_index: 0,
            }),
            vector_size,
        })
    }

    unsafe fn func(
        func: duckdb::ffi::duckdb_function_info,
        output: duckdb::ffi::duckdb_data_chunk,
    ) -> Result<()> {
        // SAFETY: registration installs these types for the lifetime of this callback.
        let bind = unsafe { &*duckdb::ffi::duckdb_function_get_bind_data(func).cast::<BindData>() };
        let init = unsafe { &*duckdb::ffi::duckdb_function_get_init_data(func).cast::<InitData>() };
        // SAFETY: this callback runs inside the live DuckDB join probe pipeline.
        let _bloom_scope = unsafe { init.blooms.enter() };
        // SAFETY: init_local installed this cursor. DuckDB gives each concurrent scan worker
        // its own local state and calls this function exclusively for that state.
        let scan = unsafe {
            duckdb::ffi::duckdb_function_get_local_init_data(func)
                .cast::<Scan>()
                .as_mut()
        }
        .ok_or("missing Vortex local scan state")?;
        loop {
            if let Some(exporter) = &mut scan.pending
                && unsafe { exporter.export(output, init.vector_size)? }
            {
                // SAFETY: the C++ scan wrapper holds its owned local state active for this callback.
                if !unsafe { filter::vp_scan_partition(scan.partition_index) } {
                    return Err("missing Vortex scan partition scope".into());
                }
                return Ok(());
            }
            scan.pending = None;
            let array = match &mut scan.stream {
                Some(stream) => bind.runtime.block_on(stream.try_next())?,
                None => None,
            };
            let Some(array) = array else {
                // Partition discovery is metadata-only; no page I/O or decode holds this lock.
                let mut partitions = init.partitions.lock();
                let partition = bind.runtime.block_on(partitions.stream.try_next())?;
                let Some(partition) = partition else {
                    // SAFETY: DuckDB lends this live output chunk exclusively to this callback.
                    unsafe { duckdb::ffi::duckdb_data_chunk_set_size(output, 0) };
                    return Ok(());
                };
                scan.partition_index = partitions.next_index;
                partitions.next_index = partitions
                    .next_index
                    .checked_add(1)
                    .ok_or("scan partition index overflow")?;
                drop(partitions);
                scan.stream = Some(partition.execute()?.boxed());
                continue;
            };
            if array.is_empty() {
                continue;
            }
            let mut ctx = bind.session.create_execution_ctx();
            scan.pending = Some(ArrayExporter::new(array, &mut ctx)?);
        }
    }
}

/// Initializes the loadable extension using the host's C API table and borrowed database.
///
/// # Safety
/// `info` and `access` must be the live handles supplied by DuckDB's extension loader.
#[cfg(feature = "loadable-extension")]
pub unsafe fn extension_init(
    info: duckdb::ffi::duckdb_extension_info,
    access: *const duckdb::ffi::duckdb_extension_access,
) -> bool {
    // SAFETY: the host supplies the access table for this call.
    let Some(access_ref) = (unsafe { access.as_ref() }) else {
        return false;
    };
    let mut success = false;
    registration::callback(
        |message| {
            if let Some(set_error) = access_ref.set_error {
                // SAFETY: the message lives through this synchronous error-reporting call.
                unsafe { set_error(info, message.as_ptr()) };
            }
        },
        || {
            unsafe extern "C" {
                fn vp_library_version() -> *const std::ffi::c_char;
            }
            // SAFETY: this stable host symbol returns a static version string without using
            // the Rust dispatch table. Reject other engines before accessing version-specific slots.
            check_version(unsafe { CStr::from_ptr(vp_library_version()) })?;
            // DuckDB 2.0 retains the V1 C API at v1.5.6; duckdb-rs uses that entrypoint and table.
            // SAFETY: the runtime was checked above and the host lends this live access table.
            if !unsafe { duckdb::ffi::duckdb_rs_extension_api_init(info, access, "v1.5.6")? } {
                return Ok(());
            }
            let get_database = access_ref
                .get_database
                .ok_or("missing DuckDB database accessor")?;
            // SAFETY: the loader lends its database for registration; never close this handle.
            let database = unsafe { get_database(info) };
            if database.is_null() {
                return Ok(());
            }
            unsafe { register(*database)? };
            success = true;
            Ok(())
        },
    );
    success
}
