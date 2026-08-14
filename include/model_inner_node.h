#ifndef HYPERCODE_MODEL_INNER_NODE_H
#define HYPERCODE_MODEL_INNER_NODE_H

#include "search_inner_node.h"
#include "leaf_node.h"
#include "configuration_search.h"
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <set>
#include <memory>
#include <set>
#include <memory>

/**
 * @class ModelInnerNode
 * @brief Inner node that uses a learned model to predict child positions
 *
 * ModelInnerNode implements a model-based inner node in the Hyper index structure.
 * It uses a linear model to predict the position of keys and maintains a list of slots
 * that can either contain direct pointers to children or duplicates with direct child pointers.
 */
class ModelInnerNode {
public:
    /**
     * @struct Slot
     * @brief Represents a slot in the model inner node with optimized memory usage
     */
    struct Slot {
        // Paper-like 16B cell: real (key,child) or duplicate child ptr.
        union {
            std::pair<KeyType, void*> KeyChildPtr;
            struct {
                void* childPtr;
                uint64_t stateBits;  // 0 => duplicate
            } dupData{};
        };

        Slot() {
            dupData.childPtr = nullptr;
            dupData.stateBits = 0;
        }
        ~Slot() {
            if (isRealChild()) {
                KeyChildPtr.~pair();
            }
        }
        bool isRealChild() const { return dupData.stateBits != 0; }
        void setRealChild(KeyType key, void* ptr) {
            new (&KeyChildPtr) std::pair<KeyType, void*>(key, ptr);
        }
        void setDuplicate(void* childPtr) {
            dupData.childPtr = childPtr;
            dupData.stateBits = 0;
        }
        void* getChildPtr() const { return dupData.childPtr; }
    };
    static_assert(sizeof(Slot) == 16, "ModelInner slot must be 16 bytes");

    /**
     * @brief Constructs a model inner node
     * @param slope Slope of the linear model
     * @param minKey Minimum key in the subtree
     * @param MR Maximum range of slots
     */
    ModelInnerNode(double slope, KeyType minKey, size_t MR);

    /**
     * @brief Destructor that properly cleans up all child pointers
     */
    ~ModelInnerNode();

    /**
     * @brief Finds the child node responsible for the given key
     * @param key Key to look up
     * @return Pointer to the child node
     */
    void* findChild(KeyType key) const;

    /**
     * @brief Associates a key with a child pointer
     * @param key Key to associate with the child
     * @param child Pointer to the child node
     */
    void setChild(KeyType key, void* child);

    /**
     * @brief Updates a child pointer where the target slot's lock is already held externally
     * @param key Key to associate with the child
     * @param child Pointer to the child node
     */
    void updateChildWithExternalLock(KeyType key, void* child);

    /**
     * @brief Associates a key with a child pointer without locks (bulk phase)
     * @param key Key to associate with the child
     * @param child Pointer to the child node
     */
    void bulkSetChild(KeyType key, void* child);

    /**
     * @brief Inserts multiple leaf nodes into the model inner node
     * @param leaves Vector of key-leaf pairs to insert
     * @param slot_counts Distribution of slots for each position
     * @param lambda Memory oversubscription factor
     */
    void bulkLoad(std::vector<std::pair<KeyType, void*>>& leaves,
                  std::vector<int>& slot_counts, double lambda);

    /**
     * @brief Gets the minimum key in the subtree
     * @return Minimum key
     */
    KeyType getMinKey() const { return minKey_; }

    /**
     * @brief Gets the slot index for a given key
     * @param key Key to get slot index for
     * @return Slot index
     */
    size_t getSlotIndex(KeyType key) const { return predictSlot(key); }

    /**
     * @brief Checks if a slot is a duplicate
     * @param idx Slot index to check
     * @return true if the slot is a duplicate, false otherwise
     */
    bool isDuplicateSlot(size_t idx) const {
        return !slots_[idx].isRealChild();
    }

    /**
     * @brief Increments the leaf node count for this subtree
     */
    void increaseLeafNodeCount();

    /**
     * @brief Gets the number of leaf nodes in this subtree
     * @return Number of leaf nodes
     */
    size_t getLeafNodeCount() const;

    /**
     * @brief Checks if the node should be rebuilt
     * @return true if a rebuild is recommended, false otherwise
     */
    bool shouldRebuild() const;

    /**
     * @brief Collect unique real children as (boundaryKey, taggedPtr), sorted by key.
     *        Used for paper §3.3.2 M-inner rebuild (no leaf data rewrite).
     */
    std::vector<std::pair<KeyType, void*>> collectSortedChildren() const;

    /**
     * @brief Null out child pointers so the destructor does not free the subtree.
     *        Call after children have been re-parented under a replacement node.
     */
    void disownChildren();

    /**
     * @brief Gets all slots in the node for efficient rebuilding
     * @return Reference to the slot vector
     */
    const std::vector<Slot>& getSlots() const { return slots_; }

    /**
     * @brief Gets the total number of slots
     * @return Number of slots
     */
    size_t getNumSlots() const { return slots_.size(); }

    /**
     * @brief Gets the child pointer at a specific index
     * @param idx Index to get child from
     * @return Pointer to the child, nullptr if invalid index
     */
    void* getChildAtIndex(size_t idx) const;

    /**
     * @brief Gets the slot lock at a specific index
     * @param idx Index of the slot
     * @return Reference to the slot's lock
     */
    HyperSlotMutex& getSlotLock(size_t idx) {
        if (slot_locks_) return slot_locks_[idx];
        static HyperSlotMutex kDummy;  // ST: no-op mutex
        return kDummy;
    }
    bool hasSlotLocks() const { return static_cast<bool>(slot_locks_); }

    uint64_t getSlotVersion(size_t idx) const {
        return slot_versions_ ? slot_versions_[idx].load(std::memory_order_acquire) : 0;
    }

    /**
     * @brief Finds child, locks the correct slot, and returns slot index atomically
     * @param key Key to find child for
     * @return Tuple of {lock, child_pointer, slot_index}
     */
    std::tuple<std::unique_lock<HyperSlotMutex>, void*, size_t> findChildWithLock(KeyType key);

    /**
    * @brief Checks if all slots are unlocked
    * @return true if all slots are unlocked, false otherwise
    */
    bool areAllSlotsUnlocked() const;

    /**
     * @brief Checks if the entire subtree is unlocked
     * @return true if all nodes in the subtree are unlocked, false otherwise
     */
    bool isSubtreeUnlocked() const;

private:
    /**
     * @brief Helper method to safely delete a child node
     * @param childPtr Tagged pointer to child node
     */
    void deleteChildNode(void* childPtr);
    
    /**
     * @brief Gets all unique child pointers (both real and duplicate)
     * @return Set of unique child pointers
     */
    std::set<void*> getAllUniqueChildren() const;

    /**
     * @brief Predicts the slot for a given key using the linear model
     * @param key Key to predict slot for
     * @return Predicted slot index
     */
    inline size_t predictSlot(KeyType key) const {
        size_t pos = ceil(slope_ * static_cast<double>(key - minKey_));
        return std::min(pos, MR_);
    };

    /**
     * @brief Reads a slot's content with seq lock protocol
     * @param idx Slot index to read
     * @param key Output parameter for the key (if real child)
     * @param child Output parameter for the child pointer
     * @param isReal Output parameter indicating if it's a real child
     * @return true if read was successful, false if should retry
     */
    bool readSlotSeqLock(size_t idx, KeyType& key, void*& child, bool& isReal) const;

    /**
     * @brief Updates a slot with seq lock protocol
     * @param idx Slot index to update
     * @param key New key value
     * @param child New child pointer
     * @param makeReal Whether to make this a real child slot
     */
    void updateSlotSeqLock(size_t idx, KeyType key, void* child, bool makeReal);

    /**
     * @brief Updates a slot as duplicate with seq lock protocol
     * @param idx Slot index to update
     * @param childPtr Direct pointer to the child node
     */
    void updateSlotDuplicateSeqLock(size_t idx, void* childPtr);

    /**
    * @brief Duplicates a slot to all slots to its right until a non-empty slot without locks (bulk phase)
    * @param from Index of the slot to duplicate
    */
    void bulkDuplicateToRight(size_t from);

    /**
     * @brief Counts leaf nodes in a subtree
     * @param node Root of the subtree
     * @return Number of leaf nodes
     */
    size_t countLeafNodesInSubtree(void* node) const;

private:
    double slope_;                       ///< Slope of the linear model
    KeyType minKey_;                     ///< Minimum key in the subtree
    size_t MR_;                          ///< Maximum range of slots
    std::vector<Slot> slots_;            ///< Array of slots for child pointers
    // MT-only side arrays (ST: zero lock/version footprint per slot).
    std::unique_ptr<HyperSlotMutex[]> slot_locks_;
    std::unique_ptr<std::atomic<uint64_t>[]> slot_versions_;

    void beginWrite(size_t idx) {
        if (slot_versions_) slot_versions_[idx].fetch_add(1, std::memory_order_release);
    }
    void endWrite(size_t idx) {
        if (slot_versions_) slot_versions_[idx].fetch_add(1, std::memory_order_release);
    }
    void fillLeadingDuplicates();

    std::atomic<size_t> initial_leaf_node_count_{0};
    std::atomic<size_t> leaf_node_count_{0};
};

#endif // HYPERCODE_MODEL_INNER_NODE_H