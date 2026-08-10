#include "../include/overflow_buffer.h"

OverflowBuffer::OverflowBuffer(size_t capacity) {
    buffer_.reserve(capacity);
}

OverflowBuffer::OverflowBuffer() = default;

OverflowBuffer::OverflowBuffer(const OverflowBuffer& other) : buffer_(other.buffer_) {}

OverflowBuffer::~OverflowBuffer() {
    buffer_.clear();
}

auto OverflowBuffer::find_position(KeyType key) {
    // Binary search to find the position where key should be
    return std::lower_bound(buffer_.begin(), buffer_.end(), key,
                            [](const auto& pair, const KeyType& k) {
                                return pair.first < k;
                            });
}

auto OverflowBuffer::find_position(KeyType key) const {
    // Binary search to find the position where key should be (const version)
    return std::lower_bound(buffer_.begin(), buffer_.end(), key,
                            [](const auto& pair, const KeyType& k) {
                                return pair.first < k;
                            });
}

void OverflowBuffer::insert(KeyType key, ValueType value) {
    // Find the position where the key should be inserted
    auto it = find_position(key);

    // If the key already exists, update its value
    if (it != buffer_.end() && it->first == key) {
        it->second = value;
    } else {
        // Insert the new key-value pair at the found position
        buffer_.emplace(it, key, value);
    }
}

OverflowBuffer* OverflowBuffer::insertRCU(KeyType key, ValueType value) const {
    // Create a new copy of the buffer
    OverflowBuffer* newBuffer = new OverflowBuffer(*this);

    // Find the position where the key should be inserted
    auto it = newBuffer->find_position(key);

    // If the key already exists, update its value
    if (it != newBuffer->buffer_.end() && it->first == key) {
        it->second = value;
    } else {
        // Insert the new key-value pair at the found position
        newBuffer->buffer_.emplace(it, key, value);
    }

    return newBuffer;
}

std::optional<ValueType> OverflowBuffer::find(KeyType key) const {
    // Find the position of the key
    auto it = find_position(key);

    // Check if the key exists
    if (it != buffer_.end() && it->first == key) {
        return it->second;
    }

    // Key not found
    return std::nullopt;
}

bool OverflowBuffer::erase(KeyType key) {
    // Find the position of the key
    auto it = find_position(key);

    // Check if the key exists
    if (it != buffer_.end() && it->first == key) {
        // Remove the key-value pair
        buffer_.erase(it);
        return true;
    }

    // Key not found
    return false;
}

OverflowBuffer* OverflowBuffer::eraseRCU(KeyType key) const {
    // Find the position of the key
    auto it = find_position(key);

    // Check if the key exists
    if (it != buffer_.end() && it->first == key) {
        // Create a new copy without the key
        OverflowBuffer* newBuffer = new OverflowBuffer(*this);
        auto new_it = newBuffer->find_position(key);
        newBuffer->buffer_.erase(new_it);
        return newBuffer;
    }

    // Key not found
    return nullptr;
}

size_t OverflowBuffer::size() const {
    return buffer_.size();
}

std::vector<std::pair<KeyType, ValueType>> OverflowBuffer::get_all() const {
    return buffer_;
}

void OverflowBuffer::bulk_load(std::vector<std::pair<KeyType, ValueType>>&& data) {
    // Move all key-value pairs and ensure they're sorted
    buffer_ = std::move(data);
}
