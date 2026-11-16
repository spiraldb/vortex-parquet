// SPDX-License-Identifier: Apache-2.0

//! Report what *encodings* a scan of a Vortex file actually produces.
//!
//! ```sh
//! cargo run --release --example encodings -- path/to/file.vortex [column ...]
//! ```
//!
//! # Why this exists
//!
//! Vortex's scan protocol hands a reader a selection mask and expects an array back, and both this
//! crate and Vortex's own `FlatLayoutReader` satisfy it by producing the whole array and *then*
//! calling `filter`. That is only cheap if the array coming out of the layout is still **encoded** —
//! filtering a dictionary's codes is not the same work as filtering the values they stand for.
//!
//! That assumption is load-bearing for the argument that Vortex does not need row-level late
//! materialization, and it was read off the code rather than measured. This measures it: it walks the
//! encoding tree of what a scan returns, so "the array stays encoded" is a fact about a file rather
//! than a belief about a design.

#![expect(clippy::expect_used, clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;

use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamExt;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::runtime::single::block_on;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::session::VortexSession;

/// Walks an array's encoding tree, counting how many nodes carry each encoding id.
///
/// Depth-first over children, because the question is not just "what is the root" — a struct of
/// dictionary-encoded strings has a canonical root and encoded leaves, and it is the leaves that
/// decide what `filter` costs.
fn walk(
    array: &ArrayRef,
    depth: usize,
    counts: &mut BTreeMap<String, usize>,
    lines: &mut Vec<String>,
) {
    let id = array.encoding_id().to_string();
    *counts.entry(id.clone()).or_default() += 1;
    if depth <= 1 {
        lines.push(format!(
            "{:indent$}{id} — {} rows, dtype {}",
            "",
            array.len(),
            array.dtype(),
            indent = depth * 2
        ));
    }
    for child in array.children() {
        walk(&child, depth + 1, counts, lines);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: encodings <path.vortex> [column ...]");
        std::process::exit(2);
    };
    let columns: Vec<String> = args.collect();

    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let fs = vortex_parquet::file::local_filesystem(&session);
        let read = fs.open_read(&path).await.expect("the file opens");
        let file = session
            .open_options()
            .open(read)
            .await
            .expect("the vortex file opens");

        let projection: Option<Expression> = if columns.is_empty() {
            None
        } else {
            let names: FieldNames = columns
                .iter()
                .map(|c| FieldName::from(c.as_str()))
                .collect();
            Some(select(names, root()))
        };

        let dtype = file.dtype().clone();
        let mut builder = ScanBuilder::new(
            session.clone(),
            file.layout_reader().expect("layout reader"),
        );
        if let Some(projection) = projection {
            builder = builder.with_projection(projection.bind(&dtype).expect("projection binds"));
        }
        let array = builder
            .into_array_stream()
            .expect("the scan builds")
            .read_all()
            .await
            .expect("the scan completes");

        let mut counts = BTreeMap::new();
        let mut lines = Vec::new();
        walk(&array, 0, &mut counts, &mut lines);

        println!("scan returned {} rows\n", array.len());
        println!("top of the encoding tree:");
        for line in &lines {
            println!("  {line}");
        }
        println!("\nencoding nodes over the whole tree:");
        for (id, n) in &counts {
            println!("  {id:<28} {n}");
        }

        // The question this example exists to answer, stated as an answer rather than left to the
        // reader: canonical encodings are the ones for which `filter` must touch every value.
        let canonical: usize = counts
            .iter()
            .filter(|(id, _)| {
                matches!(
                    id.as_str(),
                    "vortex.primitive"
                        | "vortex.varbinview"
                        | "vortex.varbin"
                        | "vortex.bool"
                        | "vortex.struct"
                        | "vortex.null"
                        | "vortex.extension"
                        | "vortex.list"
                )
            })
            .map(|(_, n)| *n)
            .sum();
        let total: usize = counts.values().sum();
        println!(
            "\n{canonical} of {total} nodes are canonical encodings ({:.0}%)",
            100.0 * canonical as f64 / total as f64
        );
    });
}
