// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prepared-set dictionary pushdown from vortex-data/vortex c3c70274c510.
//! Register through the session until the pinned Vortex revision includes this rule.

use vortex::array::ArrayRef;
use vortex::array::ArrayView;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VTable;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::Filter;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::dict::DictArrayExt;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::arrays::scalar_fn::AnyScalarFn;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::array::optimizer::kernels::ArrayKernelsExt;
use vortex::array::optimizer::rules::ArrayParentReduceRule;
use vortex::error::VortexResult;
use vortex::functions::list_contains::ListContains;
use vortex::functions::list_contains::PreparedSet;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::session::VortexSession;

pub(super) fn register(session: &VortexSession) {
    let kernels = session.kernels();
    // Scan filters execute directly, without a recursive session optimization pass.
    kernels.register_reduce_parent_rule(ListContains, Dict, &DictionaryMembershipRule);
    if !kernels.has_execute_parent(ListContains.id(), Dict.id()) {
        kernels.register_execute_parent(ListContains.id(), Dict.id(), &[execute]);
    }
    if !kernels.has_execute_parent(ListContains.id(), Filter.id()) {
        kernels.register_execute_parent(ListContains.id(), Filter.id(), &[execute_filtered]);
    }
}

#[derive(Debug)]
struct DictionaryMembershipRule;

impl ArrayParentReduceRule<Dict> for DictionaryMembershipRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Dict>,
        parent: ArrayView<'_, ScalarFn>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        reduce(array.as_ref(), parent.as_ref(), child_idx)
    }
}

fn execute_filtered(
    child: &ArrayRef,
    parent: &ArrayRef,
    child_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if child_idx != 1 {
        return Ok(None);
    }
    let Some(parent) = parent.as_opt::<ScalarFn>() else {
        return Ok(None);
    };
    let Some(options) = parent.scalar_fn().as_opt::<ListContains>() else {
        return Ok(None);
    };
    let list = parent.get_child(0);
    if PreparedSet::set_of(list).is_none() {
        return Ok(None);
    }
    // Selected page filters can produce dictionaries. Stop there so membership can probe
    // dictionary values before the default set lookup canonicalizes the selected strings.
    // Keep any canonical fallback in the new parent too, avoiding a second filter execution.
    let needles = child.clone().execute_until::<Dict>(ctx)?;
    Ok(Some(
        ListContains::try_new_opts(list.clone(), needles, *options)?.into_array(),
    ))
}

fn execute(
    child: &ArrayRef,
    parent: &ArrayRef,
    child_idx: usize,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    reduce(child, parent, child_idx)
}

fn reduce(child: &ArrayRef, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>> {
    if child_idx != 1 {
        return Ok(None);
    }
    let Some(needles) = child.as_opt::<Dict>() else {
        return Ok(None);
    };
    let Some(parent) = parent.as_opt::<ScalarFn>() else {
        return Ok(None);
    };
    let Some(options) = parent.scalar_fn().as_opt::<ListContains>() else {
        return Ok(None);
    };
    let list = parent.get_child(0);
    let Some(set) = PreparedSet::set_of(list) else {
        return Ok(None);
    };
    // Non-SQL empty membership answers false even for null codes; a dictionary would mask it.
    if set.is_empty() && !options.sql_null_semantics {
        return Ok(Some(
            ConstantArray::new(
                Scalar::bool(false, parent.dtype().nullability()),
                needles.len(),
            )
            .into_array(),
        ));
    }
    let values = needles.values();
    let codes = needles.codes();
    if values.len() > codes.len() {
        return Ok(None);
    }
    // Slicing retains the prepared set's shared allocation instead of preparing it again.
    let values =
        ListContains::try_new_opts(list.slice(0..values.len())?, values.clone(), *options)?
            .into_array()
            .optimize()?;
    // SAFETY: codes are unchanged and each original dictionary value has one result.
    let result = unsafe {
        DictArray::new_unchecked(codes.clone(), values)
            .set_all_values_referenced(needles.has_all_values_referenced())
    }
    .into_array();
    if result.dtype() == parent.dtype() {
        Ok(Some(result))
    } else {
        result.cast(parent.dtype().clone()).map(Some)
    }
}
