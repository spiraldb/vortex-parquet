// SPDX-License-Identifier: Apache-2.0

//! Scan a Parquet file from disk through the Vortex reader and report what came back.
//!
//! ```sh
//! cargo run --release --example scan_file -- path/to/file.parquet [column ...]
//! ```
//!
//! Exists because [`open_path`](vortex_parquet::open_path) is the entry point anything outside this
//! crate actually uses, and it is the one path an in-memory test cannot exercise: it goes through a
//! real filesystem, a real size probe, and the coalescing I/O driver.
//!
//! This is a smoke test with a stopwatch attached, not a benchmark. It runs once, cold, with no
//! warmup and no repetition.

#![expect(clippy::expect_used, clippy::print_stdout, clippy::print_stderr)]

use std::time::Instant;

use vortex::VortexSessionDefault;
use vortex::array::stream::ArrayStreamExt;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::io::runtime::single::block_on;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::session::VortexSession;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: scan_file <path.parquet> [column ...]");
        std::process::exit(2);
    };
    let columns: Vec<String> = args.collect();

    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);

        let opened = Instant::now();
        let reader = vortex_parquet::open_path(&path, &session)
            .await
            .expect("the file opens");
        let open_ms = opened.elapsed().as_secs_f64() * 1000.0;
        println!("opened in {open_ms:.1} ms: {} rows", reader.row_count());

        let projection: Option<Expression> = if columns.is_empty() {
            None
        } else {
            let names: FieldNames = columns
                .iter()
                .map(|c| FieldName::from(c.as_str()))
                .collect();
            Some(select(names, root()))
        };

        let scanned = Instant::now();
        let dtype = reader.dtype().clone();
        let mut builder = ScanBuilder::new(session.clone(), reader);
        if let Some(projection) = projection {
            builder = builder.with_projection(projection.bind(&dtype).expect("projection binds"));
        }
        let array = builder
            .into_array_stream()
            .expect("the scan builds")
            .read_all()
            .await
            .expect("the scan completes");
        let scan_ms = scanned.elapsed().as_secs_f64() * 1000.0;

        println!("scanned {} rows in {scan_ms:.1} ms", array.len());
        println!("result dtype: {}", array.dtype());
    })
}
