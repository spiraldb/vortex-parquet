// SPDX-License-Identifier: Apache-2.0
//! Optional join Bloom and prefix-range probes. Expressions retain IDs, never join pointers.

use std::ffi::c_void;
use std::fmt::Display;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::rc::Rc;

use duckdb::core::LogicalTypeId;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::matcher::Matcher;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::scalar_fn::Arity;
use vortex::scalar_fn::ChildName;
use vortex::scalar_fn::ExecutionArgs;
use vortex::scalar_fn::ScalarFnId;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::fill_null::FillNull;
use vortex::session::registry::CachedId;
use vortex::utils::aliases::hash_map::HashMap;

unsafe extern "C" {
    fn vp_bloom_bindings_new() -> *mut c_void;
    fn vp_bloom_bindings_free(handle: *mut c_void);
    fn vp_bloom_token() -> u64;
    fn vp_bloom_index(node: *const c_void) -> u64;
    fn vp_bloom_enter(handle: *mut c_void) -> *const c_void;
    fn vp_bloom_leave(previous: *const c_void);
    fn vp_bloom_active(token: u64, index: u64) -> bool;
    fn vp_bloom_probe(
        token: u64,
        index: u64,
        input_type: u32,
        data: *const c_void,
        len: u64,
        matches: *mut u8,
    ) -> bool;
}

pub(crate) struct Bindings(*mut c_void);
// SAFETY: owns immutable filter metadata; each probe uses its own native scratch vectors.
// Borrowed join storage is accessed only within a live matching scan callback.
unsafe impl Send for Bindings {}
// SAFETY: metadata is immutable and DuckDB's finalized join filters support concurrent reads.
unsafe impl Sync for Bindings {}
impl Bindings {
    pub(crate) fn capture() -> Self {
        // SAFETY: copies the init scope's shared_ptr, or returns null when no Bloom filters exist.
        Self(unsafe { vp_bloom_bindings_new() })
    }

    /// Caller must be inside this scan's live DuckDB callback, while its join storage is alive.
    pub(crate) unsafe fn enter(&self) -> Scope<'_> {
        let previous = if self.0.is_null() {
            None
        } else {
            // SAFETY: caller guarantees the join's lifetime; self retains native metadata.
            Some(unsafe { vp_bloom_enter(self.0) })
        };
        Scope {
            previous,
            _owner: PhantomData,
        }
    }
}
impl Drop for Bindings {
    fn drop(&mut self) {
        // SAFETY: destroys our wrapper only; metadata destruction never accesses join storage.
        unsafe { vp_bloom_bindings_free(self.0) };
    }
}

pub(crate) struct Scope<'a> {
    previous: Option<*const c_void>,
    // TLS must be restored on the entering thread, before the owner can be dropped.
    _owner: PhantomData<(&'a Bindings, Rc<()>)>,
}
impl Drop for Scope<'_> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            // SAFETY: scopes nest on one thread and retain their owning bindings.
            unsafe { vp_bloom_leave(previous) };
        }
    }
}

pub(crate) unsafe fn expression(node: *const c_void, column: Expression) -> Expression {
    // SAFETY: caller supplies a node borrowed from the current native init scope.
    let options = unsafe {
        BloomOptions {
            token: vp_bloom_token(),
            index: vp_bloom_index(node),
        }
    };
    DuckDBBloom.new_expr(options, [column])
}

#[derive(Clone)]
struct DuckDBBloom;
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BloomOptions {
    token: u64,
    index: u64,
}
impl Display for BloomOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.token, self.index)
    }
}

impl ScalarFnVTable for DuckDBBloom {
    type Options = BloomOptions;
    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("duckdb.bloom");
        *ID
    }
    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }
    fn child_name(&self, _: &Self::Options, _: usize) -> ChildName {
        ChildName::from("key")
    }
    fn return_dtype(&self, _: &Self::Options, _: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }
    fn is_strict(&self, _: &Self::Options) -> bool {
        false
    }
    fn is_infallible(&self, _: &Self::Options) -> bool {
        true
    }
    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let admit_all = || ConstantArray::new(true, args.row_count()).into_array();
        // SAFETY: this only compares IDs against TLS; no join pointer escapes the native scope.
        if !unsafe { vp_bloom_active(options.token, options.index) } {
            return Ok(admit_all());
        }
        let input = args.get(0)?;
        let rows = input.len();
        probe(input, options, ctx, &mut HashMap::default(), rows)
    }
}

struct ProbeEncodings;
impl Matcher for ProbeEncodings {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<Dict>() || array.is::<Chunked>()).then_some(array)
    }
}

// Keep source arrays alongside address keys so allocation reuse cannot alias a cached result.
// Results are scoped to this one expression execution and its immutable join filter.
type ProbeCache = HashMap<usize, (ArrayRef, ArrayRef)>;

fn probe(
    input: ArrayRef,
    options: &BloomOptions,
    ctx: &mut ExecutionCtx,
    cache: &mut ProbeCache,
    dictionary_budget: usize,
) -> VortexResult<ArrayRef> {
    let len = input.len();
    let admit_all = || ConstantArray::new(true, len).into_array();
    let kind = match input.dtype() {
        DType::Primitive(p, _) => match p {
            PType::I8 => LogicalTypeId::Tinyint,
            PType::I16 => LogicalTypeId::Smallint,
            PType::I32 => LogicalTypeId::Integer,
            PType::I64 => LogicalTypeId::Bigint,
            PType::U8 => LogicalTypeId::UTinyint,
            PType::U16 => LogicalTypeId::USmallint,
            PType::U32 => LogicalTypeId::UInteger,
            PType::U64 => LogicalTypeId::UBigint,
            _ => return Ok(admit_all()),
        },
        _ => return Ok(admit_all()),
    };
    let input = input.execute_until::<ProbeEncodings>(ctx)?;
    if let Some(chunked) = input.as_opt::<Chunked>() {
        let chunks = chunked
            .chunks()
            .iter()
            .map(|chunk| probe(chunk.clone(), options, ctx, cache, dictionary_budget))
            .collect::<VortexResult<Vec<_>>>()?;
        return Ok(
            ChunkedArray::try_new(chunks, DType::Bool(Nullability::NonNullable))?.into_array(),
        );
    }
    if let Some(dict) = input.as_opt::<Dict>() {
        let key = dict.values().addr();
        let values = if let Some((_, result)) = cache.get(&key) {
            Some(result.clone())
        } else if dict.values().len() <= dictionary_budget {
            // Pages can share a dictionary larger than one page. Compare its cost with the
            // complete input batch, and reuse the result for every page that shares its values.
            let result = probe(
                dict.values().clone(),
                options,
                ctx,
                cache,
                dict.values().len(),
            )?;
            cache.insert(key, (dict.values().clone(), result.clone()));
            Some(result)
        } else {
            // A sparse take can leave very few codes from a large dictionary; probe only those.
            None
        };
        if let Some(values) = values {
            let result = DictArray::try_new(dict.codes().clone(), values)?.into_array();
            // Null codes represent null keys, which optional join hints must admit too.
            return Ok(FillNull::try_new(result, admit_all())?.into_array());
        }
    }
    let input = input.execute::<PrimitiveArray>(ctx)?;
    let bytes = input.buffer_handle().clone().into_host_sync();
    let mut matches = vec![0u8; input.len()];
    // SAFETY: the primitive buffer has len initialized elements of the declared type.
    // Native code bounds each block, borrows the buffer synchronously, and checks scope IDs.
    let success = unsafe {
        vp_bloom_probe(
            options.token,
            options.index,
            kind as u32,
            bytes.as_ptr().cast(),
            input.len() as u64,
            matches.as_mut_ptr(),
        )
    };
    if !success {
        return Ok(admit_all());
    }
    let validity = input.validity()?.execute_mask(input.len(), ctx)?;
    let bits = if validity.all_true() {
        BitBuffer::collect_bool(input.len(), |i| matches[i] != 0)
    } else {
        // Nulls are conservatively admitted; DuckDB retains final join semantics.
        BitBuffer::collect_bool(input.len(), |i| matches[i] != 0 || !validity.value(i))
    };
    Ok(BoolArray::new(bits, Validity::NonNullable).into_array())
}
