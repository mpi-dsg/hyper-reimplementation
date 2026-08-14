#include "../include/overflow_buffer.h"
#include <stdexcept>

size_t OverflowBuffer::clamp_cap(size_t cap) {
    if (cap == 0) return kDefaultCapacity;
    if (cap > kMaxCapacity) return kMaxCapacity;
    return cap;
}

OverflowBuffer::OverflowBuffer(size_t capacity) {
    ensure_capacity(clamp_cap(capacity));
}

OverflowBuffer::OverflowBuffer() {
    ensure_capacity(kDefaultCapacity);
}

OverflowBuffer::OverflowBuffer(const OverflowBuffer& other)
        : size_(other.size_), cap_(0), pairs_(nullptr) {
    if (other.cap_ == 0) return;
    ensure_capacity(other.cap_);
    if (size_ > 0) {
        std::memcpy(pairs_, other.pairs_, size_ * sizeof(Pair));
    }
}

OverflowBuffer::~OverflowBuffer() {
    delete[] pairs_;
    pairs_ = nullptr;
    size_ = 0;
    cap_ = 0;
}

void OverflowBuffer::ensure_capacity(size_t min_cap) {
    min_cap = clamp_cap(min_cap);
    if (cap_ >= min_cap) return;
    size_t neu_cap = cap_ == 0 ? min_cap : cap_;
    while (neu_cap < min_cap) {
        size_t next = neu_cap * 2;
        if (next > kMaxCapacity || next < neu_cap) {
            neu_cap = kMaxCapacity;
            break;
        }
        neu_cap = next;
    }
    auto* neu = new Pair[neu_cap];
    if (pairs_ && size_ > 0) {
        std::memcpy(neu, pairs_, size_ * sizeof(Pair));
    }
    delete[] pairs_;
    pairs_ = neu;
    cap_ = static_cast<uint16_t>(neu_cap);
}

OverflowBuffer::Pair* OverflowBuffer::lower_bound_key(KeyType key) {
    return std::lower_bound(pairs_, pairs_ + size_, key,
                            [](const Pair& p, KeyType k) { return p.first < k; });
}

const OverflowBuffer::Pair* OverflowBuffer::lower_bound_key(KeyType key) const {
    return std::lower_bound(pairs_, pairs_ + size_, key,
                            [](const Pair& p, KeyType k) { return p.first < k; });
}

void OverflowBuffer::insert(KeyType key, ValueType value) {
    Pair* it = lower_bound_key(key);
    size_t pos = static_cast<size_t>(it - pairs_);
    if (pos < size_ && pairs_[pos].first == key) {
        pairs_[pos].second = value;
        return;
    }
    if (size_ >= kMaxCapacity) {
        // Corollary 3.1: leaf must split before capacity is exceeded. Never
        // silently drop/overwrite a different key (that creates miss rates).
        throw std::runtime_error("OverflowBuffer insert exceeds 2δ+1 capacity");
    }
    ensure_capacity(size_ + 1);
    if (pos < size_) {
        std::memmove(pairs_ + pos + 1, pairs_ + pos, (size_ - pos) * sizeof(Pair));
    }
    pairs_[pos] = {key, value};
    ++size_;
}

OverflowBuffer* OverflowBuffer::insertRCU(KeyType key, ValueType value) const {
    auto* neu = new OverflowBuffer(*this);
    neu->insert(key, value);
    return neu;
}

std::optional<ValueType> OverflowBuffer::find(KeyType key) const {
    const Pair* it = lower_bound_key(key);
    size_t pos = static_cast<size_t>(it - pairs_);
    if (pos < size_ && pairs_[pos].first == key) return pairs_[pos].second;
    return std::nullopt;
}

bool OverflowBuffer::erase(KeyType key) {
    Pair* it = lower_bound_key(key);
    size_t pos = static_cast<size_t>(it - pairs_);
    if (pos >= size_ || pairs_[pos].first != key) return false;
    if (pos + 1 < size_) {
        std::memmove(pairs_ + pos, pairs_ + pos + 1, (size_ - pos - 1) * sizeof(Pair));
    }
    --size_;
    return true;
}

OverflowBuffer* OverflowBuffer::eraseRCU(KeyType key) const {
    if (!find(key).has_value()) return nullptr;
    auto* neu = new OverflowBuffer(*this);
    neu->erase(key);
    return neu;
}

std::vector<OverflowBuffer::Pair> OverflowBuffer::get_all() const {
    return std::vector<Pair>(pairs_, pairs_ + size_);
}

void OverflowBuffer::bulk_load(std::vector<Pair>&& data) {
    size_ = 0;
    if (data.empty()) return;
    if (data.size() > kMaxCapacity) {
        throw std::runtime_error("OverflowBuffer bulk_load exceeds 2δ+1 capacity");
    }
    ensure_capacity(data.size());
    std::memcpy(pairs_, data.data(), data.size() * sizeof(Pair));
    size_ = static_cast<uint16_t>(data.size());
}
