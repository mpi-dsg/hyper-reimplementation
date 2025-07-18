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