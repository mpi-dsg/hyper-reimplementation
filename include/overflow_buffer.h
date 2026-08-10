#ifndef HYPERCODE_OVERFLOW_BUFFER_H
#define HYPERCODE_OVERFLOW_BUFFER_H

#include "common_defs.h"
#include <vector>
#include <optional>
#include <algorithm>
#include <cstring>

/**
 * @class OverflowBuffer
 * @brief Dense sorted KV overflow with small-buffer optimization (inline ≤8 pairs).
 */
class OverflowBuffer {
public:
    static constexpr size_t kInlineCap = 8;

    explicit OverflowBuffer(size_t capacity);
    OverflowBuffer();
    OverflowBuffer(const OverflowBuffer& other);
    ~OverflowBuffer();

    void insert(KeyType key, ValueType value);
    OverflowBuffer* insertRCU(KeyType key, ValueType value) const;
    std::optional<ValueType> find(KeyType key) const;
    bool erase(KeyType key);
    OverflowBuffer* eraseRCU(KeyType key) const;
    size_t size() const { return size_; }
    std::vector<std::pair<KeyType, ValueType>> get_all() const;
    /// Materialized view for callers that need a vector reference (build on demand).
    const std::vector<std::pair<KeyType, ValueType>>& data() const;
    void bulk_load(std::vector<std::pair<KeyType, ValueType>>&& data);

private:
    using Pair = std::pair<KeyType, ValueType>;

    Pair* ptr() { return heap_ ? heap_ : inline_; }
    const Pair* ptr() const { return heap_ ? heap_ : inline_; }
    size_t capacity() const { return heap_ ? heap_cap_ : kInlineCap; }
    void ensure_heap(size_t min_cap);
    void grow_for_insert();
    Pair* lower_bound_key(KeyType key);
    const Pair* lower_bound_key(KeyType key) const;

    size_t size_ = 0;
    size_t heap_cap_ = 0;
    Pair inline_[kInlineCap]{};
    Pair* heap_ = nullptr;
    // Lazy cache for data() API compatibility.
    mutable std::vector<Pair> view_cache_;
    mutable bool view_dirty_ = true;
};

#endif // HYPERCODE_OVERFLOW_BUFFER_H
