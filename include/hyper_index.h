#ifndef HYPER_H
#define HYPER_H

#include "common_defs.h"
#include "piecewise_linear_model.hpp"
#include <vector>
#include <memory>
#include <set>
#include <optional>
#include <xmmintrin.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>

// Forward declarations
class ModelInnerNode;
class SearchInnerNode;
class LeafNode;
enum class InsertResult;

/**
 * @class Hyper
 * @brief High-Performance and Memory-Efficient Learned Index
 *
 * Hyper is a learned index that combines bottom-up construction for leaf nodes with
 * top-down construction for inner nodes. It achieves high performance and memory
 * efficiency by optimizing the tradeoff between the two approaches.
 */
class Hyper {
public:
    /**
     * @brief Constructs a Hyper index
     * @param delta Error bound for piece-wise linear approximation
     * @param lambda Memory oversubscription factor
     * @param max_node_size Maximum size of a node in bytes
     */
    Hyper(double delta = 128.0, double lambda = 0.1,
          size_t max_node_size = 16*1024*1024)
            : delta_(delta), lambda_(lambda),
              max_node_size_(max_node_size) {}

    /**
     * @brief Enable/disable fine-grained locking (paper §6.1).
     *
     * Single-thread evaluations in the paper disable locking. Default is on.
     */
    static void setLockingEnabled(bool enabled) {
        setHyperLockingEnabled(enabled);
    }

    static bool lockingEnabled() { return isHyperLockingEnabled(); }

    /**
     * @brief Destructor for cleaning up the index
     */
    ~Hyper();

    /**
     * @brief Gets the root node of the index
     * @return Pointer to the root node
     */
    void* getRoot() const { return root_; }

    /**
     * @brief Bulk loads data into the index
     * @param data Vector of key-value pairs to load
     */
    void bulkLoad(const std::vector<std::pair<KeyType, ValueType>>& data);

    /**
     * @brief Finds a value for the given key
     * @param key Key to find
     * @return Optional value if found, nullopt otherwise
     */
    std::optional<ValueType> find(KeyType key) const;

    /**
     * @brief Inserts a key-value pair into the index
     * @param key Key to insert
     * @param value Value to insert
     * @throws std::runtime_error if maximum retries exceeded
     */
    void insert(KeyType key, ValueType value);

    /**
     * @brief Deletes a key from the index (Hyper paper §4.4).
     *
     * Locates the leaf, removes the entry, leaves leftmost-key metadata
     * unchanged when the deleted key was the leaf minimum, and rebuilds the
     * leaf in place if its density falls below @ref kMinLeafDensity.
     *
     * @param key Key to delete
     * @return true if the key was present and removed
     */
    bool erase(KeyType key);

    /// Lower density ratio that triggers an in-leaf rebuild after delete (§4.4).
    static constexpr double kMinLeafDensity = 0.25;

    /**
     * @brief Memory breakdown matching the paper's "index size" vs "total size"
     *        reporting (Fig. 11): index = inner structure; total includes leaves.
     *
     * Sizes are structural estimates (sizeof + heap payloads we own), not RSS.
     */
    struct MemoryStats {
        size_t index_bytes = 0;  ///< Inner nodes (model + search) and their slots
        size_t leaf_bytes = 0;   ///< Leaf nodes, overflow buffers, KV payloads
        size_t total_bytes() const { return index_bytes + leaf_bytes; }
    };

    /**
     * @brief Compute structural memory consumption of the live index tree.
     */
    MemoryStats memoryStats() const;

    /// Convenience: total structural bytes (index + leaves).
    size_t memoryBytes() const { return memoryStats().total_bytes(); }

    /**
     * @brief Inserts multiple leaf node descriptors into the index
     * @param leafDescs Vector of key-leaf pairs to insert
     */
    void insertLeafDescriptors(const std::vector<std::pair<KeyType, void*>>& leafDescs,
                               std::unique_lock<std::mutex>& parentLock);

    /**
     * @brief Executes a range query to find all key-value pairs in a range
     * @param left Lower bound of the range (inclusive)
     * @param right Upper bound of the range (inclusive)
     * @return Vector of key-value pairs in the range
     */
    std::vector<std::pair<KeyType, ValueType>> rangeQuery(KeyType left, KeyType right) const;

    /**
     * @struct PLASegment
     * @brief Represents a segment in a piece-wise linear approximation
     */
    struct PLASegment {
        KeyType min_key;   ///< Minimum key in the segment
        KeyType max_key;   ///< Maximum key in the segment
        double slope;      ///< Slope of the linear model
        size_t start_idx;  ///< Start index in the data array
        size_t end_idx;    ///< End index in the data array
    };

    /**
     * @brief Builds piece-wise linear segments for a set of data
     * @param data Vector of key-value pairs to segment
     * @return Vector of PLA segments
     */
    std::vector<PLASegment> buildPlaSegments(
            const std::vector<std::pair<KeyType, ValueType>>& data);

    /**
     * @brief Builds leaf nodes for a partition of data
     * @param data Vector of key-value pairs to build leaves for
     * @return Vector of key-leaf pairs
     */
    std::vector<std::pair<KeyType, void*>> buildLeavesForPartition(
            const std::vector<std::pair<KeyType, ValueType>>&& data);

    /**
     * @brief Builds leaf nodes for a set of data
     * @param data Vector of key-value pairs to build leaves for
     * @return Vector of key-leaf pairs
     */
    std::vector<std::pair<KeyType, void*>> buildLeaves(
            const std::vector<std::pair<KeyType, ValueType>>&& data);

    /**
     * @brief Rebuilds a model inner node
     * @param node Model inner node to rebuild
     * @param parent Parent node of the model inner node
     */
    void rebuildModelNode(ModelInnerNode* node, void* parent);

    /**
     * @brief Collects all data from a subtree
     * @param node Root of the subtree
     * @param data Vector to store collected data
     */
    void collectAllData(void* node, std::vector<std::pair<KeyType, ValueType>>& data);

    /**
     * @brief Converts a search inner node to a model inner node
     * @param sNode Search inner node to convert
     * @param parentNode Parent node of the search inner node
     */
    void convertSearchNodeToModelNode(SearchInnerNode* sNode, void* parentNode);

private:
    std::atomic<void*> root_{nullptr};        ///< Root node of the index
    std::mutex root_update_lock_;             ///< Lock only for root updates, not traversals
    double delta_;                            ///< Error bound for PLA
    double lambda_;                           ///< Memory oversubscription factor
    size_t max_node_size_;                    ///< Maximum node size in bytes
    
    // Configuration for retry behavior
    static constexpr int MAX_INSERT_RETRIES = 10;
    
    /**
     * @brief Attempt to insert a key-value pair (single attempt)
     * @param key Key to insert
     * @param value Value to insert
     * @return InsertResult indicating success or retry needed
     */
    InsertResult insertAttempt(KeyType key, ValueType value);
    
    /**
     * @brief Updates the root node using RCU
     * @param oldRoot Expected current root
     * @param newRoot New root to set
     * @return true if update succeeded, false if root changed (retry needed)
     */
    bool updateRootRCU(void* oldRoot, void* newRoot);
    
    /**
     * @brief Checks if we need to update the root during insert
     * @param currentRoot Current root pointer
     * @param leafDesc New leaf descriptor being inserted
     * @return true if root update is needed
     */
    bool needsRootUpdate(void* currentRoot, const std::pair<KeyType, void*>& leafDesc);
    
    /**
     * @brief Handles root update during split operations
     * @param oldRoot Original root before split
     * @param splitDescriptors New leaf descriptors from split
     * @return InsertResult indicating success or retry needed
     */
    InsertResult handleRootUpdate(void* oldRoot, const std::vector<std::pair<KeyType, void*>>& splitDescriptors);

    /**
     * @brief Recursively accumulate MemoryStats for a tagged subtree.
     */
    void accumulateMemory(void* node, MemoryStats& stats,
                          std::set<void*>& visited_nodes) const;
};

// Include node implementations after the Hyper class is defined
#include "model_inner_node.h"
#include "search_inner_node.h"
#include "leaf_node.h"

#endif // HYPER_H