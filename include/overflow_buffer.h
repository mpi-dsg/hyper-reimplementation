#ifndef HYPERCODE_OVERFLOW_BUFFER_H
#define HYPERCODE_OVERFLOW_BUFFER_H

#include "common_defs.h"
#include <vector>
#include <optional>
#include <algorithm>

/**
 * @class OverflowBuffer
 * @brief Stores key-value pairs in a sorted order to handle collisions
 *
 * OverflowBuffer is used when multiple keys map to the same slot in a leaf node.
 * It maintains a sorted vector of key-value pairs for efficient search.
 */
class OverflowBuffer {
public:
    /**
     * @brief Constructs an overflow buffer with a specific capacity
     * @param capacity Initial capacity of the buffer
     */
    explicit OverflowBuffer(size_t capacity);

    /**
     * @brief Default constructor
     */
    OverflowBuffer();

    /**
     * @brief Copy constructor for RCU
     */
    OverflowBuffer(const OverflowBuffer& other);

    /**
     * @brief Destructor
     */
    ~OverflowBuffer();

    /**
     * @brief Inserts a key-value pair into the buffer
     * @param key Key to insert
     * @param value Value to insert
     */
    void insert(KeyType key, ValueType value);

    /**
     * @brief Inserts a key-value pair into the buffer (creates new copy for RCU)
     * @param key Key to insert
     * @param value Value to insert
     * @return New OverflowBuffer with the inserted pair
     */
    OverflowBuffer* insertRCU(KeyType key, ValueType value) const;

    /**
     * @brief Finds a value for the given key
     * @param key Key to find
     * @return Optional value if found, nullopt otherwise
     */
    std::optional<ValueType> find(KeyType key) const;

    /**
     * @brief Removes a key-value pair from the buffer
     * @param key Key to remove
     * @return true if the key was found and removed, false otherwise
     */
    bool erase(KeyType key);

    /**
     * @brief Removes a key-value pair from the buffer (creates new copy for RCU)
     * @param key Key to remove
     * @return New OverflowBuffer without the key, or nullptr if key not found
     */
    OverflowBuffer* eraseRCU(KeyType key) const;

    /**
     * @brief Gets the number of key-value pairs in the buffer
     * @return Number of key-value pairs
     */
    size_t size() const;

    /**
     * @brief Gets all key-value pairs in the buffer
     * @return Vector of all key-value pairs
     */
    std::vector<std::pair<KeyType, ValueType>> get_all() const;

    /**
     * @brief Gets direct access to the internal buffer
     * @return Reference to the internal buffer
     */
    const std::vector<std::pair<KeyType, ValueType>>& data() const { return buffer_; }

    /**
     * @brief Loads multiple key-value pairs into the buffer using move semantics
     * @param data Vector of key-value pairs to load
     */
    void bulk_load(std::vector<std::pair<KeyType, ValueType>>&& data);

private:
    /**
     * @brief Finds the position for a key in the buffer
     * @param key Key to find position for
     * @return Iterator to the position
     */
    auto find_position(KeyType key);

    /**
     * @brief Finds the position for a key in the buffer (const version)
     * @param key Key to find position for
     * @return Const iterator to the position
     */
    auto find_position(KeyType key) const;

    std::vector<std::pair<KeyType, ValueType>> buffer_; ///< Sorted vector of key-value pairs
};

#endif // HYPERCODE_OVERFLOW_BUFFER_H
