#ifndef HYPERCODE_COMMON_DEFS_H
#define HYPERCODE_COMMON_DEFS_H

#include <cstdint>
#include <limits>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>

/**
 * Type definitions for keys and values used throughout the index
 */
using KeyType = uint64_t;    ///< 64-bit unsigned integer keys
using ValueType = uint64_t;  ///< 64-bit unsigned integer values

/**
 * @brief Global switch for fine-grained locking (paper §6.1: disabled in
 *        single-thread evaluation for fairness vs unlocked baselines).
 */
inline std::atomic<bool>& hyperLockingEnabled() {
    static std::atomic<bool> enabled{true};
    return enabled;
}

inline void setHyperLockingEnabled(bool enabled) {
    hyperLockingEnabled().store(enabled, std::memory_order_release);
}

inline bool isHyperLockingEnabled() {
    return hyperLockingEnabled().load(std::memory_order_acquire);
}

/**
 * @brief Per-slot lock that allocates a real mutex only when locking is enabled.
 *        In single-thread mode this is a null pointer (no mutex footprint).
 */
class HyperSlotMutex {
public:
    HyperSlotMutex() {
        if (isHyperLockingEnabled()) {
            mu_ = new std::mutex();
        }
    }
    ~HyperSlotMutex() { delete mu_; }
    HyperSlotMutex(const HyperSlotMutex&) = delete;
    HyperSlotMutex& operator=(const HyperSlotMutex&) = delete;

    void lock() {
        if (mu_) mu_->lock();
    }
    void unlock() {
        if (mu_) mu_->unlock();
    }
    bool try_lock() {
        return mu_ ? mu_->try_lock() : true;
    }
    bool allocated() const { return mu_ != nullptr; }

private:
    std::mutex* mu_ = nullptr;
};

/**
 * @brief Mutex guard that is a no-op when Hyper locking is disabled.
 */
class MaybeLock {
public:
    explicit MaybeLock(HyperSlotMutex& m) : lock_(m, std::defer_lock) {
        if (isHyperLockingEnabled()) {
            lock_.lock();
        }
    }

private:
    std::unique_lock<HyperSlotMutex> lock_;
};

/**
 * @enum NodeType
 * @brief Defines the possible types of nodes in the index
 *
 * Node types are stored in the lower 2 bits of pointers for type identification.
 */
enum class NodeType: uintptr_t {
    Leaf = 1,        ///< 01 - Leaf node storing key-value pairs
    SearchInner = 2, ///< 10 - Search-based inner node
    ModelInner = 3   ///< 11 - Model-based inner node
};

/**
 * @brief Tags a pointer with a node type
 * @param ptr Pointer to tag
 * @param type Node type to tag with
 * @return Tagged pointer with node type in lower bits
 */
inline void* tagPointer(void* ptr, NodeType type) {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) | static_cast<uintptr_t>(type));
}

/**
 * @brief Removes the node type tag from a pointer
 * @param ptr Tagged pointer
 * @return Original untagged pointer
 */
inline void* untagPointer(void* ptr) {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) & ~static_cast<uintptr_t>(3));
}

/**
 * @brief Extracts the node type from a tagged pointer
 * @param ptr Tagged pointer
 * @return Node type
 */
inline NodeType getNodeType(void* ptr) {
    return static_cast<NodeType>(reinterpret_cast<uintptr_t>(ptr) & 3);
}

/**
 * @brief Casts a tagged pointer to a specific node type
 * @tparam T Target type to cast to
 * @param ptr Tagged pointer to cast
 * @return Pointer of type T
 */
template <typename T>
T* taggedCast(void* ptr) {
    return reinterpret_cast<T*>(untagPointer(ptr));
}

/**
 * @brief Checks if a pointer is tagged as a leaf node
 * @param ptr Tagged pointer to check
 * @return true if the pointer is a leaf node, false otherwise
 */
inline bool isLeafNode(void* ptr) {
    return static_cast<NodeType>(reinterpret_cast<uintptr_t>(ptr) & 3) == NodeType::Leaf;
}

/**
 * @brief Checks if a pointer is tagged as a model inner node
 * @param ptr Tagged pointer to check
 * @return true if the pointer is a model inner node, false otherwise
 */
inline bool isModelInnerNode(void* ptr) {
    return static_cast<NodeType>(reinterpret_cast<uintptr_t>(ptr) & 3) == NodeType::ModelInner;
}

/**
 * @brief Checks if a pointer is tagged as a search inner node
 * @param ptr Tagged pointer to check
 * @return true if the pointer is a search inner node, false otherwise
 */
inline bool isSearchInnerNode(void* ptr) {
    return static_cast<NodeType>(reinterpret_cast<uintptr_t>(ptr) & 3) == NodeType::SearchInner;
}

#endif // HYPERCODE_COMMON_DEFS_H