#ifndef HYPERCODE_SEARCH_INNER_NODE_H
#define HYPERCODE_SEARCH_INNER_NODE_H

#include "common_defs.h"
#include <vector>
#include <algorithm>
#include <shared_mutex>

/**
 * @class SearchInnerNode
 * @brief Inner node that uses sorted key-value pairs to locate children
 *
 * SearchInnerNode is a simpler inner node implementation used when the number of
 * child pointers is small (typically <= 8). It stores child pointers in a sorted array
 * rather than using a learned model to predict positions.
 */
class SearchInnerNode {
public:
    mutable std::mutex  structural_lock_;               ///< Lock for structural changes
    struct ChildEntry {
        KeyType boundaryKey;
        void* childPtr;
        mutable std::unique_ptr<std::mutex> lock;

        ChildEntry(KeyType key, void* ptr)
                : boundaryKey(key), childPtr(ptr), lock(std::make_unique<std::mutex>()) {}

        // Copy constructor for RCU updates
        ChildEntry(const ChildEntry& other)
                : boundaryKey(other.boundaryKey), childPtr(other.childPtr),
                  lock(std::make_unique<std::mutex>()) {}

        // Move constructor
        ChildEntry(ChildEntry&& other) noexcept
                : boundaryKey(other.boundaryKey), childPtr(other.childPtr),
                  lock(std::move(other.lock)) {}

        // Copy assignment
        ChildEntry& operator=(const ChildEntry& other) {
            if (this != &other) {
                boundaryKey = other.boundaryKey;
                childPtr = other.childPtr;
                // Keep the existing lock, don't copy it
            }
            return *this;
        }
    };

    /**
     * @brief Default constructor
     */
    SearchInnerNode() = default;

    /**
     * @brief Destructor that cleans up all child pointers
     */
    ~SearchInnerNode();

    /**
     * @brief Adds a child node with the given boundary key
     * @param boundaryKey Key that defines the lower bound of this child's range
     * @param child Pointer to the child node
     */
    void addChild(KeyType boundaryKey, void* child);

    /**
     * @brief Finds the child node responsible for the given key
     * @param key Key to find child for
     * @return Pointer to the child node
     */
    void* findChild(KeyType key) const;

    /**
     * @brief Loads multiple children at once
     * @param bulkChildren Vector of key-child pairs to load
     */
    void bulk_load(const std::vector<std::pair<KeyType, void*>>& bulkChildren);

    /**
     * @brief Gets all children for conversion
     * @return Vector of key-child pairs
     */
    std::vector<std::pair<KeyType, void*>> getChildren() const;

    /**
     * @brief Gets the number of children
     * @return Number of children
     */
    size_t getChildCount() {
        return children_.size();
    }

    /**
     * @brief Finds the index of the child responsible for the given key
     * @param key Key to find child index for
     * @return Index of the child, -1 if not found
     */
    int findChildIndex(KeyType key) const;

    /**
     * @brief Gets the child at the given index
     * @param idx Index to get child from
     * @return Pointer to the child, nullptr if invalid index
     */
    void* getChildAtIndex(int idx) const;

    /**
     * @brief Gets both slot lock and child pointer atomically
     * @param idx Index of the child
     * @return Pair of {lock_pointer, child_pointer}, both nullptr if invalid index
     */
    std::pair<std::mutex*, void*> getSlotLockAndChild(int idx);

    /**
     * @brief Gets the version counter for consistent reads
     * @return Current version
     */
    uint64_t getVersion() const;

    /**
     * @brief Increments version for writes
     */
    void incrementVersion();

    /**
    * @brief Checks if all slots are unlocked
    * @return true if all slots are unlocked, false otherwise
    */
    bool areAllSlotsUnlocked() const;

private:
    /**
     * @brief Reads with version checking for consistency
     * @param operation Lambda that performs the read operation
     * @return Result of the operation
     */
    template<typename Operation>
    auto readWithVersionCheck(Operation&& op) const -> decltype(op()) {
        uint64_t version_before, version_after;
        decltype(op()) result;

        do {
            version_before = getVersion();
            // Skip if write is in progress (odd version)
            if (version_before & 1) {
                std::this_thread::yield();
                continue;
            }

            result = op();

            version_after = getVersion();
        } while (version_before != version_after);

        return result;
    }

private:
    std::vector<ChildEntry> children_;              ///< Sorted vector of children with locks
    std::atomic<uint64_t> version_{0};              ///< Version counter for consistent reads
};

#endif // HYPERCODE_SEARCH_INNER_NODE_H