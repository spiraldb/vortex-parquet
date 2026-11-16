// SPDX-License-Identifier: Apache-2.0
// Adapted from vortex-duckdb's ExternalVectorBuffer and primitive/string exporters.
#include "duckdb.hpp"

using namespace duckdb;

// DuckDB 2.0 stores data and cardinality in the buffer. A slice buffer retains the Rust
// owner through its parent, following vortex-duckdb's ExternalVectorBuffer.
class ParquetExternalBuffer final : public StandardVectorBuffer {
    shared_ptr<ParquetExternalBuffer> parent;
    void *owner = nullptr;
    void (*release)(void *) = nullptr;
public:
    ParquetExternalBuffer(void *owner, void (*release)(void *))
        : StandardVectorBuffer(nullptr, count_t(0), 0), owner(owner), release(release) {}
    ParquetExternalBuffer(shared_ptr<ParquetExternalBuffer> parent, data_ptr_t data, idx_t count, idx_t width)
        : StandardVectorBuffer(data, count_t(count), width), parent(std::move(parent)) {}
    ~ParquetExternalBuffer() override { if (release) release(owner); }
};

extern "C" {
void *vp_buffer_new(void *owner, void (*release)(void *)) {
    return new shared_ptr<ParquetExternalBuffer>(make_shared_ptr<ParquetExternalBuffer>(owner, release));
}
void vp_buffer_free(void *buffer) {
    delete reinterpret_cast<shared_ptr<ParquetExternalBuffer> *>(buffer);
}
void vp_vector_buffer(void *vector, void *buffer, void *data, uint64_t count, uint64_t width) {
    auto &target = *reinterpret_cast<Vector *>(vector);
    auto &owner = *reinterpret_cast<shared_ptr<ParquetExternalBuffer> *>(buffer);
    target.SetBuffer(make_buffer<ParquetExternalBuffer>(owner, static_cast<data_ptr_t>(data), count, width));
}
void vp_string_buffer(void *vector, void *buffer) {
    buffer_ptr<VectorBuffer> owner = *reinterpret_cast<shared_ptr<ParquetExternalBuffer> *>(buffer);
    StringVector::AddAuxiliaryData(*reinterpret_cast<Vector *>(vector), make_uniq<VectorBufferHolder>(std::move(owner)));
}
bool vp_vector_reference(void *vector, void *source) noexcept {
    // Reference rejects different logical types even if their physical storage is identical.
    // Keep its C++ exception inside this boundary so Rust can report a query error.
    try {
        reinterpret_cast<Vector *>(vector)->Reference(*reinterpret_cast<Vector *>(source));
        return true;
    } catch (...) { return false; }
}
void *vp_dictionary_new(void *source, uint64_t size) noexcept {
    try {
        auto &values = *reinterpret_cast<Vector *>(source);
        values.BufferMutable().SetVectorSize(size);
        auto buffer = DictionaryVector::CreateReusableDictionary(values.GetType(), size);
        buffer->data.Reference(values);
        return new buffer_ptr<DictionaryEntry>(std::move(buffer));
    } catch (...) { return nullptr; }
}
void vp_dictionary_free(void *dictionary) {
    delete reinterpret_cast<buffer_ptr<DictionaryEntry> *>(dictionary);
}
void *vp_dictionary_clone(void *dictionary) noexcept {
    try {
        return new buffer_ptr<DictionaryEntry>(
            *reinterpret_cast<buffer_ptr<DictionaryEntry> *>(dictionary));
    } catch (...) { return nullptr; }
}
bool vp_vector_dictionary(void *vector, void *dictionary, void *selection, uint64_t count) noexcept {
    try {
        auto &target = *reinterpret_cast<Vector *>(vector);
        auto &buffer = *reinterpret_cast<buffer_ptr<DictionaryEntry> *>(dictionary);
        if (target.GetType() != buffer->data.GetType()) return false;
        target.Dictionary(buffer, *reinterpret_cast<SelectionVector *>(selection), count);
        return true;
    } catch (...) { return false; }
}
void vp_vector_constant(void *vector) {
    reinterpret_cast<Vector *>(vector)->SetVectorType(VectorType::CONSTANT_VECTOR);
}
}
