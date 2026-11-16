// SPDX-License-Identifier: Apache-2.0
//! Build a small C++ API layer against upstream DuckDB headers, following vortex-duckdb.

use std::env;
use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use flate2::read::GzDecoder;
use sha2::Digest;
use sha2::Sha256;

const VERSION: &str = "v2.0.0";
const SOURCE_URL: &str = "https://github.com/duckdb/duckdb/archive/refs/tags/v2.0.0.tar.gz";
const SOURCE_SHA256: &str = "7e89053255046bfb26afbd7f524a2fb84df7dbef496aa911ea35e4a1182fdc00";
const SOURCES: &[&str] = &["cpp/registration.cpp", "cpp/vector.cpp", "cpp/filter.cpp"];

fn duckdb_source() -> Result<PathBuf, Box<dyn Error>> {
    let source = env::var_os("DUCKDB_SOURCE_DIR");
    let version = env::var("DUCKDB_VERSION").ok();
    if source.is_some() != version.is_some() {
        return Err("set DUCKDB_SOURCE_DIR and DUCKDB_VERSION together".into());
    }
    if let Some(source) = source {
        if version.as_deref() != Some(VERSION) {
            return Err(format!("this adapter requires DUCKDB_VERSION={VERSION}").into());
        }
        let source = PathBuf::from(source).canonicalize()?;
        if !source.join("src/include/duckdb.hpp").is_file() {
            return Err("DUCKDB_SOURCE_DIR must contain DuckDB's src/include directory".into());
        }
        println!("cargo:rerun-if-changed={}/src/include", source.display());
        println!(
            "cargo:rerun-if-changed={}/third_party/snappy",
            source.display()
        );
        return Ok(source);
    }

    // Keep downloads in Cargo's build directory, never in the source tree.
    let cache = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?)
        .join(format!("duckdb-source-{VERSION}"));
    let marker = cache.join(".complete");
    let source = cache.join("duckdb-2.0.0");
    if fs::read_to_string(&marker).ok().as_deref() == Some(SOURCE_SHA256)
        && source.join("src/include/duckdb.hpp").is_file()
    {
        return Ok(source);
    }

    println!("cargo:warning=Downloading upstream DuckDB {VERSION} headers");
    let timeout = env::var("CARGO_HTTP_TIMEOUT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(90);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout))
        .build()?;
    let bytes = client.get(SOURCE_URL).send()?.error_for_status()?.bytes()?;
    let mut digest = String::with_capacity(64);
    for byte in Sha256::digest(&bytes) {
        write!(digest, "{byte:02x}")?;
    }
    if digest != SOURCE_SHA256 {
        return Err(format!("DuckDB source checksum mismatch: {digest}").into());
    }

    // An interrupted extraction cannot be mistaken for a complete cached download.
    if cache.exists() {
        fs::remove_dir_all(&cache)?;
    }
    fs::create_dir_all(&cache)?;
    let mut archive = tar::Archive::new(GzDecoder::new(bytes.as_ref()));
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path.starts_with("duckdb-2.0.0/src/include")
            || path.starts_with("duckdb-2.0.0/third_party/snappy")
            || path == Path::new("duckdb-2.0.0/LICENSE")
        {
            entry.unpack_in(&cache)?;
        }
    }
    if !source.join("src/include/duckdb.hpp").is_file() {
        return Err("DuckDB source archive did not contain the required headers".into());
    }
    fs::write(marker, SOURCE_SHA256)?;
    Ok(source)
}

fn main() -> Result<(), Box<dyn Error>> {
    for key in ["DUCKDB_SOURCE_DIR", "DUCKDB_VERSION", "CARGO_HTTP_TIMEOUT"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rerun-if-changed=cpp/include");
    for source in SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-changed=cpp/snappy.cpp");
    let source = duckdb_source()?;
    let mut build = cc::Build::new();
    build.cpp(true);
    let include = source.join("src/include");
    let snappy = source.join("third_party/snappy");
    if build.get_compiler().is_like_msvc() {
        build.include(&include).include(&snappy);
    } else {
        // Upstream headers are dependencies, not code whose warnings this crate owns.
        build
            .flag("-isystem")
            .flag(&include)
            .flag("-isystem")
            .flag(&snappy);
    }
    build
        .cpp(true)
        .std("c++20")
        .define("NDEBUG", None)
        .include("cpp/include")
        .files(SOURCES)
        .flag_if_supported("-fvisibility=hidden")
        .compile("vortex_parquet_extras");
    // DuckDB 2.0 keeps Snappy inside its Parquet extension rather than exporting it from
    // libduckdb. Build the pinned upstream decoder privately; namespace it to avoid host clashes.
    let mut snappy_build = cc::Build::new();
    // Upstream leaves SIMD disabled for portable builds. Cargo's target features tell us
    // when NEON is guaranteed, so the private decoder can use its vector byte-shuffle path.
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64")
        && env::var("CARGO_CFG_TARGET_FEATURE")
            .is_ok_and(|features| features.split(',').any(|feature| feature == "neon"))
    {
        snappy_build.define("SNAPPY_HAVE_NEON", "1");
    }
    snappy_build
        .cpp(true)
        .std("c++20")
        .opt_level(3)
        .warnings(false)
        .define("NDEBUG", None)
        .define("duckdb_snappy", "vortex_parquet_snappy")
        .include(&snappy)
        .file("cpp/snappy.cpp")
        .file(snappy.join("snappy.cc"))
        .file(snappy.join("snappy-sinksource.cc"))
        .file(snappy.join("snappy-stubs-internal.cc"))
        .flag_if_supported("-fvisibility=hidden")
        .compile("vortex_parquet_snappy");
    if env::var_os("CARGO_FEATURE_LOADABLE_EXTENSION").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
    Ok(())
}
