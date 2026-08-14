#ifndef HYPERCODE_OVERFLOW_BUFFER_H
#define HYPERCODE_OVERFLOW_BUFFER_H

#include "common_defs.h"
#include <vector>
#include <optional>
#include <algorithm>
#include <cstring>

/**
 * @class OverflowBuffer
 * @brief Dense sorted KV overflow with a flat array (no std::vector on hot path).
 *
 * Paper Corollary 3.1: conflicts at a slot are bounded by 2δ+1 (δ=128 → 257).
 * Storage is a single contiguous allocation grown up to that cap.
 */
class OverflowBuffer {
public:
    using Pair = std::pair<KeyType, ValueType>;

    /// Hard cap = 2δ+1 with paper δ=128.
    static constexpr size_t kMaxCapacity = 257;
    /// Default initial allocation for collision of two keys.
    static constexpr size_t kDefaultCapacity = 8;

    explicit OverflowBuffer(size_t capacity);
    OverflowBuffer();
    OverflowBuffer(const OverflowBuffer& other);
    OverflowBuffer& operator=(const OverflowBuffer&) = delete;
    ~OverflowBuffer();

    void insert(KeyType key, ValueType value);
    OverflowBuffer* insertRCU(KeyType key, ValueType value) const;
    std::optional<ValueType> find(KeyType key) const;
    bool erase(KeyType key);
    OverflowBuffer* eraseRCU(KeyType key) const;

    size_t size() const { return size_; }
    size_t capacity() const { return cap_; }
    bool empty() const { return size_ == 0; }

    Pair* data() { return pairs_; }
    const Pair* data() const { return pairs_; }
    Pair* begin() { return pairs_; }
    Pair* end() { return pairs_ + size_; }
    const Pair* begin() const { return pairs_; }
    const Pair* end() const { return pairs_ + size_; }
    const Pair& front() const { return pairs_[0]; }

    std::vector<Pair> get_all() const;
    void bulk_load(std::vector<Pair>&& data);

private:
    Pair* lower_bound_key(KeyType key);
    const Pair* lower_bound_key(KeyType key) const;
    void ensure_capacity(size_t min_cap);
    static size_t clamp_cap(size_t cap);

    uint16_t size_ = 0;
    uint16_t cap_ = 0;
    Pair* pairs_ = nullptr;
};

#endif // HYPERCODE_OVERFLOW_BUFFER_H
