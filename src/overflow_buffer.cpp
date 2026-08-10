#include "../include/overflow_buffer.h"
#include <new>

OverflowBuffer::OverflowBuffer(size_t capacity) {
    if (capacity > kInlineCap) {
        ensure_heap(capacity);
    }
}

OverflowBuffer::OverflowBuffer() = default;

OverflowBuffer::OverflowBuffer(const OverflowBuffer& other) : size_(other.size_) {
    if (other.heap_) {
        ensure_heap(other.heap_cap_);
        std::memcpy(heap_, other.heap_, size_ * sizeof(Pair));
    } else {
        std::memcpy(inline_, other.inline_, size_ * sizeof(Pair));
    }
    view_dirty_ = true;
}

OverflowBuffer::~OverflowBuffer() {
    delete[] heap_;
    heap_ = nullptr;
}

void OverflowBuffer::ensure_heap(size_t min_cap) {
    if (heap_ && heap_cap_ >= min_cap) return;
    size_t cap = std::max(min_cap, heap_ ? heap_cap_ * 2 : kInlineCap * 2);
    Pair* neu = new Pair[cap];
    const Pair* src = ptr();
    if (size_ > 0) std::memcpy(neu, src, size_ * sizeof(Pair));
    delete[] heap_;
    heap_ = neu;
    heap_cap_ = cap;
}

void OverflowBuffer::grow_for_insert() {
    if (size_ < capacity()) return;
    ensure_heap(size_ + 1);
}

OverflowBuffer::Pair* OverflowBuffer::lower_bound_key(KeyType key) {
    Pair* b = ptr();
    return std::lower_bound(b, b + size_, key,
                            [](const Pair& p, KeyType k) { return p.first < k; });
}

const OverflowBuffer::Pair* OverflowBuffer::lower_bound_key(KeyType key) const {
    const Pair* b = ptr();
    return std::lower_bound(b, b + size_, key,
                            [](const Pair& p, KeyType k) { return p.first < k; });
}

void OverflowBuffer::insert(KeyType key, ValueType value) {
    Pair* it = lower_bound_key(key);
    size_t pos = static_cast<size_t>(it - ptr());
    if (pos < size_ && ptr()[pos].first == key) {
        ptr()[pos].second = value;
        view_dirty_ = true;
        return;
    }
    grow_for_insert();
    Pair* base = ptr();
    if (pos < size_) {
        std::memmove(base + pos + 1, base + pos, (size_ - pos) * sizeof(Pair));
    }
    base[pos] = {key, value};
    ++size_;
    view_dirty_ = true;
}

OverflowBuffer* OverflowBuffer::insertRCU(KeyType key, ValueType value) const {
    auto* neu = new OverflowBuffer(*this);
    neu->insert(key, value);
    return neu;
}

std::optional<ValueType> OverflowBuffer::find(KeyType key) const {
    const Pair* it = lower_bound_key(key);
    size_t pos = static_cast<size_t>(it - ptr());
    if (pos < size_ && ptr()[pos].first == key) return ptr()[pos].second;
    return std::nullopt;
}

bool OverflowBuffer::erase(KeyType key) {
    Pair* it = lower_bound_key(key);
    size_t pos = static_cast<size_t>(it - ptr());
    if (pos >= size_ || ptr()[pos].first != key) return false;
    Pair* base = ptr();
    if (pos + 1 < size_) {
        std::memmove(base + pos, base + pos + 1, (size_ - pos - 1) * sizeof(Pair));
    }
    --size_;
    view_dirty_ = true;
    return true;
}

OverflowBuffer* OverflowBuffer::eraseRCU(KeyType key) const {
    if (!find(key).has_value()) return nullptr;
    auto* neu = new OverflowBuffer(*this);
    neu->erase(key);
    return neu;
}

std::vector<std::pair<KeyType, ValueType>> OverflowBuffer::get_all() const {
    return std::vector<Pair>(ptr(), ptr() + size_);
}

const std::vector<std::pair<KeyType, ValueType>>& OverflowBuffer::data() const {
    if (view_dirty_) {
        view_cache_.assign(ptr(), ptr() + size_);
        view_dirty_ = false;
    }
    return view_cache_;
}

void OverflowBuffer::bulk_load(std::vector<std::pair<KeyType, ValueType>>&& data) {
    size_ = 0;
    view_dirty_ = true;
    if (data.empty()) return;
    if (data.size() <= kInlineCap && !heap_) {
        std::memcpy(inline_, data.data(), data.size() * sizeof(Pair));
        size_ = data.size();
        return;
    }
    ensure_heap(data.size());
    std::memcpy(heap_, data.data(), data.size() * sizeof(Pair));
    size_ = data.size();
}
