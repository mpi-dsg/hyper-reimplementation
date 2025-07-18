#ifndef HYPERCODE_LEAF_NODE_H
#define HYPERCODE_LEAF_NODE_H

#include "overflow_buffer.h"
#include "hyper_index.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <optional>
#include <mutex>

/**
 * MSB_MASK is used to distinguish between key-value pairs and pointers
 * The most significant bit is set to 1 for key-value pairs
 * and 0 for pointers to overflow buffers
 */
constexpr KeyType MSB_MASK = 1ULL << 63;

/**
 * @enum InsertResult
 * @brief Result codes for insert operations
 */
enum class InsertResult {
    Success,         // Insert succeeded without split
    SuccessWithSplit, // Insert succeeded and triggered a split
    RetryFromRoot    // Key exceeds leaf bounds, need to retry from root
};

/**
 * @struct InsertReturn
 * @brief Return value for insert operations containing result code and optional split descriptors
 */
struct InsertReturn {
    InsertResult result;
    std::optional<std::vector<std::pair<KeyType, void*>>> splitDescriptors;
    
    InsertReturn(InsertResult res) : result(res) {}
    InsertReturn(InsertResult res, std::vector<std::pair<KeyType, void*>>&& splits) 
        : result(res), splitDescriptors(std::move(splits)) {}
};

/**
 * @class LeafNode
 * @brief Leaf nodes store key-value pairs and manage overflow buffers with slot-level locking.
 *
 * LeafNode is responsible for storing key-value pairs using a linear model to predict
 * slot positions. When multiple keys map to the same position, overflow buffers are used
 * to handle collisions. Each slot has its own mutex for fine-grained locking.
 */
class LeafNode {
public:
    /**
     * @struct Slot
     * @brief Represents a slot in the leaf node that can contain either a key-value pair
     * or a pointer to an overflow buffer, with its own mutex for concurrency control.
     */
    struct Slot {
        /**
         * @union SlotData
         * @brief Union to store either a key-value pair or a pointer to an overflow buffer.
         */
        union SlotData {
            struct {
                std::atomic<KeyType> key;     // Key with MSB set to 1
                ValueType value; // Associated value
            } kv;
            std::atomic<OverflowBuffer*> overflowPtr; // Pointer to overflow buffer with MSB of 0
        } data;

        mutable std::mutex lock;  // Mutex for this specific slot

        Slot();
        ~Slot();

        /**
         * @brief Destroys the slot content properly based on its type
         */
        void destroy();

        /**
         * @brief Sets the slot to contain a single key-value pair
         * @param k Key to store
         * @param v Value to store
         */
        void setSingle(KeyType k, ValueType v);

        /**
         * @brief Creates an overflow buffer and adds the key-value pair
         * @param k Key to store
         * @param v Value to store
         */
        void setOverflow(KeyType k, ValueType v);

        /**
         * @brief Checks if the slot contains a key-value pair
         * @return true if the slot contains a key-value pair, false otherwise
         */
        bool isKV() const;

        /**
         * @brief Checks if the slot contains a pointer to an overflow buffer
         * @return true if the slot contains a pointer, false otherwise
         */
        bool isPointer() const;

        /**
         * @brief Checks if the slot is empty
         * @return true if the slot is empty, false otherwise
         */
        bool isEmpty() const;
    };

    /**
     * @brief Constructs a leaf node with a linear model for key placement
     * @param slope Slope of the linear model
     * @param minKey Minimum key in the node
     * @param maxKey Maximum key in the node
     */
    LeafNode(double slope, KeyType minKey, KeyType maxKey);

    /**
     * @brief Copy constructor for RCU
     */
    LeafNode(const LeafNode& other);

    /**
     * @brief Destructor that properly cleans up all slots
     */
    ~LeafNode();

    /**
     * @brief Get the slots in the leaf node
     * @return Reference to the vector of slots
     */
    const std::vector<Slot>& getSlots() const { return slots_; }

    /**
     * @brief Insert a key-value pair into the leaf node with slot-level locking
     * @param key Key to insert
     * @param value Value to insert
     * @param delta Error bound for PLA
     * @return InsertReturn indicating success, split, or retry needed
     */
    InsertReturn insert(
            KeyType key,
            ValueType value,
            double delta);

    /**
     * @brief Find a value for the given key (lock-free read)
     * @param key Key to look for
     * @return Optional value if found, nullopt otherwise
     */
    std::optional<ValueType> find(KeyType key) const;

    /**
     * @brief Remove a key-value pair from the leaf node
     * @param key Key to erase
     * @return true if the key was found and removed, false otherwise
     */
    bool erase(KeyType key);

    /**
     * @brief Load multiple key-value pairs into the leaf node
     * @param data Vector of key-value pairs to load
     */
    void bulkLoad(std::vector<std::pair<KeyType, ValueType>>&& data);

    /**
     * @brief Collect all key-value pairs from the leaf node and its overflow buffers
     * @return Vector of all key-value pairs
     */
    std::vector<std::pair<KeyType, ValueType>> gatherAll() const;

    /**
     * @brief Get the minimum key in the leaf node
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
     * @brief Decodes a stored key by removing the MSB flag if appropriate
     * @param stored_key Key with possible MSB set
     * @return Original key value
     */
    KeyType decodeKey(KeyType stored_key) const;

    /**
    * @brief Set the maximum possible key for this leaf node
    * @param maxKey New maximum possible key
    */
    void setMaxPossibleKey(KeyType maxKey) { maxPossibleKey_ = maxKey; }

    /**
     * @brief Perform split operation with parent lock already held
     * @param delta Error bound for PLA
     * @return Optional vector of leaf descriptors if split successful, nullopt if aborted
     */
    std::optional<std::vector<std::pair<KeyType, void*>>> performSplitWithParentLock(double delta);

    /**
     * @brief Check if any slot in the leaf node is currently locked
     * @return true if any slot is locked, false if all slots are free
     */
    bool hasAnySlotLocked() const;

    /**
     * @brief Try to acquire all slot locks (for split operations)
     * @return vector of unique_locks if successful, empty vector if any lock fails
     */
    std::vector<std::unique_lock<std::mutex>> tryLockAllSlots();

private:
    /**
     * @brief Predicts the slot position for a given key using the linear model
     * @param key Key to predict slot for
     * @return Predicted slot index
     */
    inline size_t predictSlot(KeyType key) const {
        size_t pos = ceil(slope_ * static_cast<double>(key - minKey_));
        return std::min(pos, MR_);
    };

    /**
     * @brief Checks if a significant distribution shift has occurred (Policy Two)
     * @return true if a redistribution is needed, false otherwise
     */
    bool checkPolicyTwo();

    /**
     * @brief Performs the actual split operation with all locks held
     * @param data All data from the leaf node
     * @param delta Error bound for PLA
     * @return Vector of new leaf descriptors
     */
    std::vector<std::pair<KeyType, void*>> performSplit(
            const std::vector<std::pair<KeyType, ValueType>>& data,
            double delta);

    double slope_;                   // Slope of the linear model
    size_t MR_;                      // Maximum range of slots
    KeyType minKey_;                 // Minimum key in the node
    KeyType maxPossibleKey_;         // Maximum possible key for this node
    std::vector<Slot> slots_;        // Array of slots for storing data

    std::atomic<uint32_t> op_counter_;            // Counter for operations to trigger Policy Two
    std::vector<int> init_histogram_; // Initial key distribution histogram
};

#endif // HYPERCODE_LEAF_NODE_H