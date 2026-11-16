// SPDX-License-Identifier: Apache-2.0
// Native scan hooks missing from the DuckDB C API, tied to DuckDB 2.0.0.
#include "duckdb.hpp"
#include "duckdb/main/capi/capi_internal_table.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"
#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/optimizer/filter_combiner.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "vortex_parquet.hpp"
#include <atomic>
#include <mutex>

using namespace duckdb;

extern "C" bool vp_supports_filter_column(const void *bind_data, uint64_t column);
extern "C" bool vp_column_bounds(const void *bind_data, uint64_t column,
                                  duckdb_value *minimum, duckdb_value *maximum);

namespace {
std::atomic<uint64_t> next_bloom_token {1};
struct JoinFilter {
    bool prefix_range;
    unique_ptr<FunctionData> data;
    double acceptance = -1;
    bool enabled = true;
};
struct BloomBindings {
    uint64_t token = next_bloom_token.fetch_add(1, std::memory_order_relaxed);
    vector<JoinFilter> filters;
};
thread_local const shared_ptr<BloomBindings> *init_blooms = nullptr;
thread_local const BloomBindings *active_blooms = nullptr;

struct FilterNode {
    uint32_t kind;
    uint32_t comparison = 0;
    Value value;
    shared_ptr<DynamicFilterData> dynamic;
    uint64_t bloom_index = 0;
    string field;
    vector<unique_ptr<FilterNode>> children;
    explicit FilterNode(uint32_t kind) : kind(kind) {}
};

bool SupportedStringExpression(const Expression &expr) {
    if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) return false;
    auto &f = expr.Cast<BoundFunctionExpression>();
    if (f.Function().GetName() != "contains" && f.Function().GetName() != "prefix" && f.Function().GetName() != "suffix") return false;
    if (f.GetChildren().size() != 2 || f.IsVolatile() || expr.GetReturnType().id() != LogicalTypeId::BOOLEAN) return false;
    auto &column = *f.GetChildren()[0];
    auto &constant = *f.GetChildren()[1];
    if (column.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF &&
        column.GetExpressionClass() != ExpressionClass::BOUND_REF) return false;
    if (column.GetReturnType().id() != LogicalTypeId::VARCHAR ||
        !StringType::GetCollation(column.GetReturnType()).empty()) return false;
    return constant.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT &&
           constant.GetReturnType().id() == LogicalTypeId::VARCHAR &&
           !constant.Cast<BoundConstantExpression>().GetValue().IsNull();
}

string StringPattern(const BoundFunctionExpression &f) {
    // Convert a literal string search into LIKE, escaping its metacharacters first.
    auto &literal = StringValue::Get(f.GetChildren()[1]->Cast<BoundConstantExpression>().GetValue());
    string pattern;
    if (f.Function().GetName() != "prefix") pattern += '%';
    for (auto c : literal) {
        if (c == '%' || c == '_' || c == '\\') pattern += '\\';
        pattern += c;
    }
    if (f.Function().GetName() != "suffix") pattern += '%';
    return pattern;
}

bool MembershipColumn(const Expression &expr, const LogicalGet &get, const void *bind, optional_idx &column) {
    if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) return false;
    // Nested fields support their own filters, but the scalar bridge cannot own a nested constant.
    if (expr.GetReturnType().IsNested()) return false;
    auto &ref = expr.Cast<BoundColumnRefExpression>();
    if (ref.Depth() || ref.Binding().table_index != get.table_index ||
        ref.Binding().column_index >= get.GetColumnIds().size()) return false;
    auto &index = get.GetColumnIds()[ref.Binding().column_index];
    if (!index.HasPrimaryIndex() || index.IsVirtualColumn() || index.HasChildren() || index.IsPushdownExtract()) return false;
    auto primary = index.GetPrimaryIndex();
    if (!vp_supports_filter_column(bind, primary)) return false;
    if (expr.GetReturnType().id() == LogicalTypeId::VARCHAR && !StringType::GetCollation(expr.GetReturnType()).empty()) return false;
    const auto projection = ref.Binding().column_index.GetIndex();
    if (column.IsValid() && column.GetIndex() != projection) return false;
    column = projection;
    return true;
}

unique_ptr<TableFilter> ExactMembership(const Expression &expr, const LogicalGet &get,
                                      const void *bind, optional_idx &column) {
    if (expr.GetExpressionType() == ExpressionType::COMPARE_IN &&
        expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
        auto &op = expr.Cast<BoundOperatorExpression>();
        if (op.GetChildren().size() < 2 || !MembershipColumn(*op.GetChildren()[0], get, bind, column)) return nullptr;
        vector<Value> values;
        for (idx_t i = 1; i < op.GetChildren().size(); ++i) {
            if (op.GetChildren()[i]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) return nullptr;
            auto &value = op.GetChildren()[i]->Cast<BoundConstantExpression>().GetValue();
            if (value.IsNull() || value.type() != op.GetChildren()[0]->GetReturnType()) return nullptr;
            values.push_back(value);
        }
        return make_uniq<ExpressionFilter>(ExpressionFilter::CreateInExpression(
            make_uniq<BoundReferenceExpression>(op.GetChildren()[0]->GetReturnType(), 0), std::move(values)));
    }
    if (expr.GetExpressionType() != ExpressionType::CONJUNCTION_OR ||
        expr.GetExpressionClass() != ExpressionClass::BOUND_CONJUNCTION) return nullptr;
    auto &op = expr.Cast<BoundConjunctionExpression>();
    if (op.GetChildren().empty()) return nullptr;
    vector<unique_ptr<Expression>> comparisons;
    for (auto &child : op.GetChildren()) {
        if (!BoundComparisonExpression::IsComparison(*child)) return nullptr;
        auto &cmp = child->Cast<BoundFunctionExpression>();
        const Expression *lhs = &BoundComparisonExpression::Left(cmp), *rhs = &BoundComparisonExpression::Right(cmp);
        auto comparison = cmp.GetExpressionType();
        if (lhs->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
            std::swap(lhs, rhs);
            comparison = FlipComparisonExpression(comparison);
        }
        switch (comparison) {
        case ExpressionType::COMPARE_EQUAL:
        case ExpressionType::COMPARE_NOTEQUAL:
        case ExpressionType::COMPARE_LESSTHAN:
        case ExpressionType::COMPARE_GREATERTHAN:
        case ExpressionType::COMPARE_LESSTHANOREQUALTO:
        case ExpressionType::COMPARE_GREATERTHANOREQUALTO: break;
        default: return nullptr;
        }
        if (!MembershipColumn(*lhs, get, bind, column) ||
            rhs->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) return nullptr;
        auto &value = rhs->Cast<BoundConstantExpression>().GetValue();
        if (value.IsNull() || value.type() != lhs->GetReturnType()) return nullptr;
        comparisons.push_back(BoundComparisonExpression::Create(comparison,
            make_uniq<BoundReferenceExpression>(lhs->GetReturnType(), 0), make_uniq<BoundConstantExpression>(value)));
    }
    return make_uniq<ExpressionFilter>(BoundConjunctionExpression::Create(ExpressionType::CONJUNCTION_OR, std::move(comparisons)));
}

void OwnLargeMembership(ClientContext &context, LogicalGet &get, FunctionData *data,
                        vector<unique_ptr<Expression>> &expressions) {
    if (!data || get.table_filters.HasFilters()) return;
    auto &bind = data->Cast<CTableBindData>();
    // Upstream 1843ce957305: retain residuals on small scans to preserve join planning.
    if (!bind.stats || !bind.stats->has_estimated_cardinality || bind.stats->estimated_cardinality < (1ULL << 20)) return;
    TableFilterSet owned;
    vector<idx_t> owned_indices;
    for (idx_t i = 0; i < expressions.size(); ++i) {
        optional_idx column;
        auto filter = ExactMembership(*expressions[i], get, bind.bind_data, column);
        if (filter) {
            owned.PushFilter(ProjectionIndex(column.GetIndex()), std::move(filter));
            owned_indices.push_back(i);
        }
    }
    if (!owned.HasFilters()) return;
    // Installing table_filters bypasses DuckDB's later combiner pass, so use that same
    // combiner for the remaining expressions here. Keep the original plan on contradiction.
    FilterCombiner combiner(context);
    for (idx_t i = 0; i < expressions.size(); ++i) {
        if (std::find(owned_indices.begin(), owned_indices.end(), i) != owned_indices.end()) continue;
        if (combiner.AddFilter(expressions[i]->Copy()) == FilterResult::UNSATISFIABLE) return;
    }
    vector<FilterPushdownResult> results;
    get.table_filters = combiner.GenerateTableScanFilters(get.GetColumnIds(), results);
    vector<unique_ptr<Expression>> remaining;
    combiner.GenerateFilters([&](unique_ptr<Expression> expr) { remaining.push_back(std::move(expr)); });
    for (idx_t i = 0; i < remaining.size(); ++i) {
        auto &expr = *remaining[i];
        if (i < results.size() && results[i] != FilterPushdownResult::NO_PUSHDOWN) continue;
        if (expr.IsVolatile() || (expr.CanThrow() &&
            (expr.GetExpressionType() == ExpressionType::COMPARE_IN || remaining.size() > 1))) continue;
        if (combiner.TryPushdownGenericExpression(get, expr) == FilterPushdownResult::PUSHED_DOWN_FULLY) {
            remaining.erase_at(i);
            if (i < results.size()) results.erase_at(i);
            --i;
        }
    }
    for (auto &entry : owned) get.table_filters.PushFilter(entry.GetIndex(), entry.TakeFilter());
    expressions = std::move(remaining);
}

// Stable adapter wire kinds, independent of DuckDB's removed legacy filter classes.
enum FilterKind : uint32_t {
    COMPARISON = 0, IS_NULL = 1, IS_NOT_NULL = 2, OR = 3, AND = 4, STRUCT = 5,
    OPTIONAL = 6, BLOOM = 10, NO_FILTER = 100, STRING_PATTERN = 101, DYNAMIC = 102, MEMBERSHIP = 103, ALWAYS_FALSE = 104
};

unique_ptr<FilterNode> OnColumn(const Expression &column, unique_ptr<FilterNode> filter) {
    vector<StructExtractPathEntry> path;
    auto &base = PeelStructExtractPath(column, path);
    if (base.GetExpressionClass() != ExpressionClass::BOUND_REF || base.Cast<BoundReferenceExpression>().Index() != 0) {
        throw InternalException("Unsupported Vortex filter subject");
    }
    for (auto entry = path.rbegin(); entry != path.rend(); ++entry) {
        auto parent = make_uniq<FilterNode>(STRUCT);
        parent->field = entry->child_name.GetIdentifierName();
        parent->children.push_back(std::move(filter));
        filter = std::move(parent);
    }
    return filter;
}

unique_ptr<FilterNode> Comparison(const Expression &lhs, const Expression &rhs, ExpressionType comparison) {
    switch (comparison) {
    case ExpressionType::COMPARE_EQUAL:
    case ExpressionType::COMPARE_NOTEQUAL:
    case ExpressionType::COMPARE_LESSTHAN:
    case ExpressionType::COMPARE_GREATERTHAN:
    case ExpressionType::COMPARE_LESSTHANOREQUALTO:
    case ExpressionType::COMPARE_GREATERTHANOREQUALTO: break;
    default: throw InternalException("Unsupported Vortex comparison");
    }
    if (rhs.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
        throw InternalException("Vortex comparison requires a constant");
    }
    auto node = make_uniq<FilterNode>(COMPARISON);
    node->comparison = static_cast<uint32_t>(comparison);
    node->value = rhs.Cast<BoundConstantExpression>().GetValue();
    return OnColumn(lhs, std::move(node));
}

unique_ptr<FilterNode> ConvertExpression(const Expression &expr, BloomBindings &blooms) {
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT && expr.GetReturnType().id() == LogicalTypeId::BOOLEAN) {
        const auto &value = expr.Cast<BoundConstantExpression>().GetValue();
        return make_uniq<FilterNode>(!value.IsNull() && value.GetValue<bool>() ? NO_FILTER : ALWAYS_FALSE);
    }
    if (ExpressionFilter::IsRootOptionalExpression(expr)) {
        auto node = make_uniq<FilterNode>(OPTIONAL);
        auto child = ExpressionFilter::GetOptionalFilterChild(expr);
        if (!child) return make_uniq<FilterNode>(NO_FILTER);
        try {
            node->children.push_back(ConvertExpression(*child, blooms));
        } catch (const std::exception &) {
            // Unsupported optional hints never remove rows; the host checks the predicate.
            return make_uniq<FilterNode>(NO_FILTER);
        }
        return node;
    }
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
        auto node = make_uniq<FilterNode>(expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND ? AND : OR);
        for (auto &child : expr.Cast<BoundConjunctionExpression>().GetChildren()) {
            node->children.push_back(ConvertExpression(*child, blooms));
        }
        return node;
    }
    if (BoundComparisonExpression::IsComparison(expr)) {
        auto &cmp = expr.Cast<BoundFunctionExpression>();
        const Expression *lhs = &BoundComparisonExpression::Left(cmp), *rhs = &BoundComparisonExpression::Right(cmp);
        auto comparison = cmp.GetExpressionType();
        if (lhs->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
            std::swap(lhs, rhs);
            comparison = FlipComparisonExpression(comparison);
        }
        return Comparison(*lhs, *rhs, comparison);
    }
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
        auto &op = expr.Cast<BoundOperatorExpression>();
        auto &children = op.GetChildren();
        if (children.size() == 1 && (expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ||
                                    expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL)) {
            return OnColumn(*children[0], make_uniq<FilterNode>(
                expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ? IS_NULL : IS_NOT_NULL));
        }
        if (expr.GetExpressionType() == ExpressionType::COMPARE_IN && children.size() > 1) {
            auto node = make_uniq<FilterNode>(MEMBERSHIP);
            for (idx_t i = 1; i < children.size(); ++i) {
                if (children[i]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
                    throw InternalException("Vortex membership requires constants");
                }
                auto value = make_uniq<FilterNode>(COMPARISON);
                value->value = children[i]->Cast<BoundConstantExpression>().GetValue();
                node->children.push_back(std::move(value));
            }
            return OnColumn(*children[0], std::move(node));
        }
    }
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
        auto &f = expr.Cast<BoundFunctionExpression>();
        // Statistics can simplify a comparison to TRUE for non-null rows while retaining
        // its null semantics as constant_or_null(true, column, constant).
        if (f.Function().GetName() == "constant_or_null" && !f.GetChildren().empty() &&
            f.GetReturnType().id() == LogicalTypeId::BOOLEAN &&
            f.GetChildren()[0]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
            const auto &value = f.GetChildren()[0]->Cast<BoundConstantExpression>().GetValue();
            if (value.IsNull() || !value.GetValue<bool>()) return make_uniq<FilterNode>(ALWAYS_FALSE);
            auto node = make_uniq<FilterNode>(AND);
            for (idx_t i = 1; i < f.GetChildren().size(); ++i) {
                auto &child = *f.GetChildren()[i];
                if (child.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
                    if (child.Cast<BoundConstantExpression>().GetValue().IsNull()) return make_uniq<FilterNode>(ALWAYS_FALSE);
                } else {
                    node->children.push_back(OnColumn(child, make_uniq<FilterNode>(IS_NOT_NULL)));
                }
            }
            return node;
        }
        if (SupportedStringExpression(expr)) {
            auto node = make_uniq<FilterNode>(STRING_PATTERN);
            node->value = Value(StringPattern(f));
            return OnColumn(*f.GetChildren()[0], std::move(node));
        }
        if (expr.GetExpressionType() == ExpressionType::COMPARE_BETWEEN && BoundBetweenExpression::HasValidBindData(f)) {
            auto node = make_uniq<FilterNode>(AND);
            node->children.push_back(Comparison(BoundBetweenExpression::Input(f), BoundBetweenExpression::LowerBound(f),
                                                BoundBetweenExpression::LowerComparisonType(f)));
            node->children.push_back(Comparison(BoundBetweenExpression::Input(f), BoundBetweenExpression::UpperBound(f),
                                                BoundBetweenExpression::UpperComparisonType(f)));
            return node;
        }
        if (f.BindInfo() && f.GetChildren().size() == 1) {
            if (f.Function().GetName() == DynamicFilterScalarFun::NAME) {
                auto node = make_uniq<FilterNode>(DYNAMIC);
                node->dynamic = f.BindInfo()->Cast<DynamicFilterFunctionData>().filter_data;
                if (!node->dynamic) return make_uniq<FilterNode>(NO_FILTER);
                node->comparison = static_cast<uint32_t>(node->dynamic->comparison_type);
                return OnColumn(*f.GetChildren()[0], std::move(node));
            }
            if (f.Function().GetName() == BloomFilterScalarFun::NAME ||
                f.Function().GetName() == PrefixRangeScalarFun::NAME) {
                auto node = make_uniq<FilterNode>(BLOOM);
                node->bloom_index = blooms.filters.size();
                blooms.filters.push_back({f.Function().GetName() == PrefixRangeScalarFun::NAME,
                                          f.BindInfo()->Copy()});
                return OnColumn(*f.GetChildren()[0], std::move(node));
            }
        }
    }
    throw InternalException("Unsupported Vortex filter expression: %s", expr.ToString());
}

struct FilterRoot {
    idx_t column;
    unique_ptr<FilterNode> filter;
};
using FilterRoots = vector<FilterRoot>;
thread_local const FilterRoots *current_filters = nullptr;
thread_local const TableFunctionInitInput *current_input = nullptr;
std::atomic<table_function_init_global_t> original_init {nullptr};

// Keep DuckDB's C API local state intact inside a wrapper that records the current row-group
// ordinal. No private C API state layout is inspected. The Rust callback reports its ordinal
// synchronously through active_scan; the engine later reads it from this owned local state.
struct ScanLocalState : public LocalTableFunctionState {
    explicit ScanLocalState(unique_ptr<LocalTableFunctionState> inner_p) : inner(std::move(inner_p)) {}
    unique_ptr<LocalTableFunctionState> inner;
    idx_t batch_index = 0;
};
std::atomic<table_function_init_local_t> original_local_init {nullptr};
std::atomic<table_function_t> original_function {nullptr};
thread_local ScanLocalState *active_scan = nullptr;

unique_ptr<LocalTableFunctionState> InitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                            GlobalTableFunctionState *global) {
    return make_uniq<ScanLocalState>(original_local_init.load()(context, input, global));
}

void ScanFunction(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
    auto &local = input.local_state->Cast<ScanLocalState>();
    struct Scope {
        TableFunctionInput &input;
        optional_ptr<LocalTableFunctionState> previous_local;
        ScanLocalState *previous_active;
        Scope(TableFunctionInput &input_p, ScanLocalState &local)
            : input(input_p), previous_local(input.local_state), previous_active(active_scan) {
            input.local_state = local.inner.get();
            active_scan = &local;
        }
        ~Scope() {
            input.local_state = previous_local;
            active_scan = previous_active;
        }
    } scope(input, local);
    original_function.load()(context, input, output);
}

// Optional join ranges are hints: the join still checks every key. Avoid copying almost the
// entire column for broad hints. The range fraction is a cost estimate, never a correctness
// assumption about the value distribution; missing exact footer bounds keep the filter.
bool IsOptionalRange(const FilterNode &node) {
    if (node.kind != OPTIONAL || node.children.size() != 1) return false;
    const auto &child = *node.children[0];
    if (child.kind != COMPARISON) return false;
    auto comparison = static_cast<ExpressionType>(child.comparison);
    return comparison == ExpressionType::COMPARE_LESSTHAN || comparison == ExpressionType::COMPARE_LESSTHANOREQUALTO ||
           comparison == ExpressionType::COMPARE_GREATERTHAN || comparison == ExpressionType::COMPARE_GREATERTHANOREQUALTO;
}

bool HasOptionalRange(const FilterNode &node) {
    if (IsOptionalRange(node)) return true;
    for (const auto &child : node.children) if (HasOptionalRange(*child)) return true;
    return false;
}

void OmitBroadRanges(FilterNode &node, const Value &minimum, const Value &maximum) {
    if (IsOptionalRange(node)) {
        const auto &child = *node.children[0];
        if (child.value.type() != minimum.type() || child.value.IsNull()) return;
        const auto low = minimum.GetValue<double>();
        const auto high = maximum.GetValue<double>();
        if (high <= low) return;
        const auto bound = child.value.GetValue<double>();
        const auto comparison = static_cast<ExpressionType>(child.comparison);
        const bool lower = comparison == ExpressionType::COMPARE_GREATERTHAN ||
                           comparison == ExpressionType::COMPARE_GREATERTHANOREQUALTO;
        const auto fraction = (lower ? high - bound : bound - low) / (high - low);
        // DuckDB likewise pauses optional min/max filtering at 75% observed acceptance.
        if (fraction >= 0.75) {
            node.kind = 100;
            node.children.clear();
        }
        return;
    }
    for (auto &child : node.children) OmitBroadRanges(*child, minimum, maximum);
}

double PrefixRangeAcceptance(const JoinFilter &binding, const Value &minimum, const Value &maximum) noexcept {
    try {
        const auto &filter = binding.data->Cast<PrefixRangeFunctionData>();
        if (!filter.filter || !filter.filter->IsInitialized() ||
            minimum.type() != filter.key_type || maximum.type() != filter.key_type) return -1;
        constexpr idx_t samples = 128;
        const auto low = minimum.GetValue<double>();
        const auto high = maximum.GetValue<double>();
        Vector keys(filter.key_type, samples);
        for (idx_t i = 0; i < samples; ++i) {
            keys.SetValue(i, Value::DOUBLE(low + (high - low) * (double(i) / (samples - 1)))
                                .DefaultCastAs(filter.key_type));
        }
        SelectionVector selection(samples);
        return double(filter.filter->LookupKeys(keys, selection, samples)) / samples;
    } catch (...) {
        return -1;
    }
}

unique_ptr<GlobalTableFunctionState> Init(ClientContext &context, TableFunctionInitInput &input) {
    FilterRoots filters;
    auto blooms = make_shared_ptr<BloomBindings>();
    if (input.filters) {
        if (input.filters->HasMultiColumnFilters()) throw InternalException("Unexpected Vortex multi-column filter");
        for (auto &entry : *input.filters) {
            if (entry.GetIndex().GetIndex() >= input.column_ids.size()) throw InternalException("Vortex filter column out of bounds");
            const auto column = input.column_ids[entry.GetIndex()];
            auto &filter = ExpressionFilter::GetExpressionFilter(entry.Filter(), "Vortex scan");
            const auto first_binding = blooms->filters.size();
            auto converted = ConvertExpression(*filter.expr, *blooms);
            if (input.bind_data && (HasOptionalRange(*converted) || first_binding < blooms->filters.size())) {
                duckdb_value minimum = nullptr, maximum = nullptr;
                if (vp_column_bounds(input.bind_data->Cast<CTableBindData>().bind_data, column, &minimum, &maximum)) {
                    unique_ptr<Value> min(reinterpret_cast<Value *>(minimum));
                    unique_ptr<Value> max(reinterpret_cast<Value *>(maximum));
                    OmitBroadRanges(*converted, *min, *max);
                    for (idx_t i = first_binding; i < blooms->filters.size(); ++i) {
                        auto &binding = blooms->filters[i];
                        if (binding.prefix_range) binding.acceptance = PrefixRangeAcceptance(binding, *min, *max);
                    }
                }
            }
            filters.push_back({column, std::move(converted)});
        }
    }
    // Optional join hints must earn the cost of materializing compressed keys. Use the most
    // selective estimated prefix range, leaving all joins to check the other keys. Sampling
    // affects cost alone: it never excludes a row. Missing estimates retain their hints.
    optional_idx best_prefix;
    for (idx_t i = 0; i < blooms->filters.size(); ++i) {
        const auto &binding = blooms->filters[i];
        if (!binding.prefix_range || binding.acceptance < 0 || binding.acceptance >= 0.75) continue;
        if (!best_prefix.IsValid() || binding.acceptance < blooms->filters[best_prefix.GetIndex()].acceptance) {
            best_prefix = i;
        }
    }
    for (idx_t i = 0; i < blooms->filters.size(); ++i) {
        auto &binding = blooms->filters[i];
        if (binding.prefix_range && binding.acceptance >= 0) {
            binding.enabled = best_prefix.IsValid() && i == best_prefix.GetIndex();
        }
    }
    struct Scope {
        const FilterRoots *previous;
        const TableFunctionInitInput *previous_input;
        const shared_ptr<BloomBindings> *previous_blooms;
        Scope(const FilterRoots *filters, const TableFunctionInitInput *input,
              const shared_ptr<BloomBindings> *blooms)
            : previous(current_filters), previous_input(current_input), previous_blooms(init_blooms) {
            current_filters = filters;
            current_input = input;
            init_blooms = blooms;
        }
        ~Scope() {
            current_filters = previous;
            current_input = previous_input;
            init_blooms = previous_blooms;
        }
    } scope(&filters, &input, &blooms);
    // The original C API initializer invokes Rust synchronously. Snapshots are borrowed only
    // during that callback; Rust creates owned Vortex expressions before this scope ends.
    return original_init.load()(context, input);
}
}

extern "C" {
bool vp_scan_partition(uint64_t index) noexcept {
    if (!active_scan) return false;
    active_scan->batch_index = index;
    return true;
}
void *vp_bloom_bindings_new() {
    if (!init_blooms || (*init_blooms)->filters.empty()) return nullptr;
    return new shared_ptr<BloomBindings>(*init_blooms);
}
void vp_bloom_bindings_free(void *handle) {
    delete static_cast<shared_ptr<BloomBindings> *>(handle);
}
uint64_t vp_bloom_token() { return init_blooms ? (*init_blooms)->token : 0; }
uint64_t vp_bloom_index(const void *node) { return static_cast<const FilterNode *>(node)->bloom_index; }
const void *vp_bloom_enter(void *handle) {
    auto previous = active_blooms;
    active_blooms = static_cast<shared_ptr<BloomBindings> *>(handle)->get();
    return previous;
}
void vp_bloom_leave(const void *previous) {
    active_blooms = static_cast<const BloomBindings *>(previous);
}
bool vp_bloom_active(uint64_t token, uint64_t index) {
    return active_blooms && active_blooms->token == token && index < active_blooms->filters.size() &&
           active_blooms->filters[index].enabled;
}
bool vp_bloom_probe(uint64_t token, uint64_t index, uint32_t input_type,
                    const void *data, uint64_t len, uint8_t *matches) noexcept {
    // Bloom copies retain shared storage; prefix-range metadata borrows the join's storage.
    // Both are probed only inside that join's live scan callback. Expressions carry IDs alone
    // and decline the hint if another scan drives their task.
    if (!vp_bloom_active(token, index)) return false;
    try {
        LogicalType type;
        switch (input_type) {
        case DUCKDB_TYPE_TINYINT: type = LogicalType::TINYINT; break;
        case DUCKDB_TYPE_SMALLINT: type = LogicalType::SMALLINT; break;
        case DUCKDB_TYPE_INTEGER: type = LogicalType::INTEGER; break;
        case DUCKDB_TYPE_BIGINT: type = LogicalType::BIGINT; break;
        case DUCKDB_TYPE_UTINYINT: type = LogicalType::UTINYINT; break;
        case DUCKDB_TYPE_USMALLINT: type = LogicalType::USMALLINT; break;
        case DUCKDB_TYPE_UINTEGER: type = LogicalType::UINTEGER; break;
        case DUCKDB_TYPE_UBIGINT: type = LogicalType::UBIGINT; break;
        default: return false;
        }
        auto &binding = active_blooms->filters[index];
        const PrefixRangeFilter *prefix_range = nullptr;
        const BloomFilter *bloom = nullptr;
        if (binding.prefix_range) {
            auto &filter = binding.data->Cast<PrefixRangeFunctionData>();
            if (type != filter.key_type || !filter.filter || !filter.filter->IsInitialized()) return false;
            prefix_range = filter.filter.get();
        } else {
            auto &filter = binding.data->Cast<BloomFilterFunctionData>();
            // Hash using exactly the join key's physical type. Unsupported casts admit all rows.
            if (type != filter.key_type || !filter.filter || !filter.filter->IsSealed()) return false;
            bloom = filter.filter.get();
        }
        Vector hashes(LogicalType::HASH);
        SelectionVector selection(STANDARD_VECTOR_SIZE);
        const auto width = GetTypeIdSize(type.InternalType());
        for (idx_t offset = 0; offset < len; offset += STANDARD_VECTOR_SIZE) {
            const auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, len - offset);
            auto bytes = static_cast<const uint8_t *>(data) + offset * width;
            Vector keys(type, const_cast<data_ptr_t>(bytes), count);
            idx_t selected;
            if (prefix_range) {
                selected = prefix_range->LookupKeys(keys, selection, count);
            } else {
                VectorOperations::Hash(keys, hashes, count);
                selected = bloom->LookupHashes(hashes, selection, count);
            }
            for (idx_t i = 0; i < selected; ++i) {
                matches[offset + selection.get_index(i)] = 1;
            }
        }
        return true;
    } catch (...) {
        // Rust discards partial output on false: a failed optional hint must admit all rows.
        return false;
    }
}
void vp_enable_filters(duckdb_table_function handle) {
    auto &function = *reinterpret_cast<TableFunction *>(handle);
    auto expected = static_cast<table_function_init_global_t>(nullptr);
    original_init.compare_exchange_strong(expected, function.init_global);
    function.init_global = Init;
    auto expected_local = static_cast<table_function_init_local_t>(nullptr);
    original_local_init.compare_exchange_strong(expected_local, function.init_local);
    auto expected_function = static_cast<table_function_t>(nullptr);
    original_function.compare_exchange_strong(expected_function, function.function);
    function.init_local = InitLocal;
    function.function = ScanFunction;
    function.get_partition_data = [](ClientContext &, TableFunctionGetPartitionInput &input) {
        if (input.partition_info.RequiresPartitionColumns()) {
            throw NotImplementedException("Vortex scans report batch indexes, not column partitions");
        }
        return OperatorPartitionData(input.local_state->Cast<ScanLocalState>().batch_index);
    };
    function.filter_pushdown = true;
    function.pushdown_complex_filter = OwnLargeMembership;
    function.filter_prune = true;
    function.statistics = [](ClientContext &, const FunctionData *bind, column_t column) -> unique_ptr<BaseStatistics> {
        if (!bind) return nullptr;
        duckdb_value minimum = nullptr, maximum = nullptr;
        if (!vp_column_bounds(bind->Cast<CTableBindData>().bind_data, column, &minimum, &maximum)) return nullptr;
        // Rust transfers these owned C API Values. RAII also releases them if stats construction throws.
        unique_ptr<Value> min(reinterpret_cast<Value *>(minimum));
        unique_ptr<Value> max(reinterpret_cast<Value *>(maximum));
        auto stats = make_uniq<BaseStatistics>(NumericStats::CreateUnknown(min->type()));
        NumericStats::SetMin(*stats, *min);
        NumericStats::SetMax(*stats, *max);
        return stats;
    };
    // DuckDB selects this virtual column for count(*) instead of requesting a real column.
    function.get_virtual_columns = [](ClientContext &, optional_ptr<FunctionData>) {
        virtual_column_map_t columns;
        columns.emplace(COLUMN_IDENTIFIER_EMPTY, TableColumn("", LogicalType::BOOLEAN));
        return columns;
    };
    function.pushdown_expression = [](ClientContext &, const LogicalGet &, Expression &expr) {
        return SupportedStringExpression(expr);
    };
    function.supports_pushdown_type = [](const FunctionData &bind, idx_t column) {
        return vp_supports_filter_column(bind.Cast<CTableBindData>().bind_data, column);
    };
}
uint64_t vp_projection_count() {
    if (!current_input) return 0;
    return current_input->projection_ids.empty() ? current_input->column_ids.size() : current_input->projection_ids.size();
}
bool vp_is_empty_column(uint64_t column) { return column == COLUMN_IDENTIFIER_EMPTY; }
uint64_t vp_projection_column(uint64_t i) {
    auto index = current_input->projection_ids.empty() ? i : current_input->projection_ids[i];
    return current_input->column_ids[index];
}
uint64_t vp_filter_count() { return current_filters ? current_filters->size() : 0; }
const void *vp_filter_get(uint64_t i, uint64_t *column) {
    auto &root = (*current_filters)[i];
    *column = root.column;
    return root.filter.get();
}
uint32_t vp_filter_kind(const void *node) { return static_cast<const FilterNode *>(node)->kind; }
uint32_t vp_filter_comparison(const void *node) { return static_cast<const FilterNode *>(node)->comparison; }
void *vp_filter_dynamic(const void *node) {
    return new shared_ptr<DynamicFilterData>(static_cast<const FilterNode *>(node)->dynamic);
}
void vp_dynamic_free(void *handle) { delete static_cast<shared_ptr<DynamicFilterData> *>(handle); }
duckdb_value vp_dynamic_value(void *handle) noexcept {
    // A missing or unavailable hint must admit all rows. Never propagate a C++ exception into Rust.
    try {
        auto &data = *static_cast<shared_ptr<DynamicFilterData> *>(handle);
        annotated_shared_lock<annotated_shared_mutex> lock(data->lock);
        if (!data->initialized) return nullptr;
        return reinterpret_cast<duckdb_value>(new Value(data->constant));
    } catch (...) { return nullptr; }
}
uint64_t vp_filter_children(const void *node) { return static_cast<const FilterNode *>(node)->children.size(); }
const void *vp_filter_child(const void *node, uint64_t i) { return static_cast<const FilterNode *>(node)->children[i].get(); }
const char *vp_filter_field(const void *node, uint64_t *len) {
    auto &field = static_cast<const FilterNode *>(node)->field;
    *len = field.size();
    return field.data();
}
duckdb_value vp_filter_value(const void *node) {
    return reinterpret_cast<duckdb_value>(const_cast<Value *>(&static_cast<const FilterNode *>(node)->value));
}
uint32_t vp_value_type(duckdb_value value) {
    switch (reinterpret_cast<Value *>(value)->type().id()) {
    case LogicalTypeId::BOOLEAN: return DUCKDB_TYPE_BOOLEAN;
    case LogicalTypeId::TINYINT: return DUCKDB_TYPE_TINYINT;
    case LogicalTypeId::SMALLINT: return DUCKDB_TYPE_SMALLINT;
    case LogicalTypeId::INTEGER: return DUCKDB_TYPE_INTEGER;
    case LogicalTypeId::BIGINT: return DUCKDB_TYPE_BIGINT;
    case LogicalTypeId::UTINYINT: return DUCKDB_TYPE_UTINYINT;
    case LogicalTypeId::USMALLINT: return DUCKDB_TYPE_USMALLINT;
    case LogicalTypeId::UINTEGER: return DUCKDB_TYPE_UINTEGER;
    case LogicalTypeId::UBIGINT: return DUCKDB_TYPE_UBIGINT;
    case LogicalTypeId::VARCHAR: return DUCKDB_TYPE_VARCHAR;
    case LogicalTypeId::BLOB: return DUCKDB_TYPE_BLOB;
    case LogicalTypeId::DECIMAL: return DUCKDB_TYPE_DECIMAL;
    case LogicalTypeId::DATE: return DUCKDB_TYPE_DATE;
    case LogicalTypeId::TIME: return DUCKDB_TYPE_TIME;
    case LogicalTypeId::TIME_NS: return DUCKDB_TYPE_TIME_NS;
    case LogicalTypeId::TIMESTAMP: return DUCKDB_TYPE_TIMESTAMP;
    case LogicalTypeId::TIMESTAMP_SEC: return DUCKDB_TYPE_TIMESTAMP_S;
    case LogicalTypeId::TIMESTAMP_MS: return DUCKDB_TYPE_TIMESTAMP_MS;
    case LogicalTypeId::TIMESTAMP_NS: return DUCKDB_TYPE_TIMESTAMP_NS;
    case LogicalTypeId::TIMESTAMP_TZ: return DUCKDB_TYPE_TIMESTAMP_TZ;
    default: return DUCKDB_TYPE_INVALID;
    }
}
const char *vp_value_bytes(duckdb_value value, uint64_t *len) {
    auto &bytes = StringValue::Get(*reinterpret_cast<Value *>(value));
    *len = bytes.size();
    return bytes.data();
}
}
