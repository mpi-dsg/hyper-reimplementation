#include "../include/model_inner_node.h"
#include "../include/epoch_manager.h"

// --- ModelInnerNode implementation ---
ModelInnerNode::ModelInnerNode(double slope, KeyType minKey, size_t MR)
        : slope_(slope), minKey_(minKey), MR_(MR), slots_(MR + 1) {
}

ModelInnerNode::~ModelInnerNode() {
    // Get all unique children and delete them to avoid memory leaks
    auto uniqueChildren = getAllUniqueChildren();
    
    for (void* childPtr : uniqueChildren) {
        deleteChildNode(childPtr);
    }
}

bool ModelInnerNode::readSlotSeqLock(size_t idx, KeyType& key, void*& child, bool& isReal) const {
    uint64_t version_before, version_after;

    do {
        // Read version before accessing data
        version_before = slots_[idx].version.load(std::memory_order_acquire);

        // If version is odd, a writer is active - retry
        if (version_before & 1) {
            continue;
        }

        // Read slot data
        isReal = slots_[idx].isRealChild();
        if (isReal) {
            key = slots_[idx].KeyChildPtr.first;
            child = slots_[idx].KeyChildPtr.second;
        } else {
            // For duplicate slots, we have the direct child pointer
            key = 0; // Not used for duplicates
            child = slots_[idx].getChildPtr();
        }

        // Read version after accessing data
        version_after = slots_[idx].version.load(std::memory_order_acquire);

    } while (version_before != version_after || (version_before & 1));

    return true;
}

void ModelInnerNode::updateSlotSeqLock(size_t idx, KeyType key, void* child, bool makeReal) {
    // Caller must already hold Lock

    // Begin write operation (increment to odd number)
    slots_[idx].beginWrite();

    // Update slot data
    if (makeReal) {
        if (!slots_[idx].isRealChild()) {
            // Converting from duplicate to real child
            slots_[idx].setRealChild(key, child);
        } else {
            // Updating existing real child - safely delete the old one
            // void* oldChild = slots_[idx].KeyChildPtr.second;
            // if (oldChild && oldChild != child) {
            //     safeDelete(oldChild);
            // }
            slots_[idx].KeyChildPtr.first = key;
            slots_[idx].KeyChildPtr.second = child;
        }
    }

    // End write operation (increment to even number)
    slots_[idx].endWrite();
}

void ModelInnerNode::updateSlotDuplicateSeqLock(size_t idx, void* childPtr) {
    // Caller must already hold Lock

    // Begin write operation (increment to odd number)
    slots_[idx].beginWrite();

    // Update slot as duplicate with direct child pointer
    slots_[idx].setDuplicate(childPtr);

    // End write operation (increment to even number)
    slots_[idx].endWrite();
}

void* ModelInnerNode::findChild(KeyType key) const {
    // Predict slot position for the key
    size_t idx = predictSlot(key);

    KeyType slotKey;
    void* slotChild;
    bool isReal;

    // Read the predicted slot with seq lock
    readSlotSeqLock(idx, slotKey, slotChild, isReal);

    if (isReal) {
        // Check if we need to look at the previous slot
        if (slotKey > key && idx > 0) {
            idx--;
            readSlotSeqLock(idx, slotKey, slotChild, isReal);
        }

        if (isReal) {
            return slotChild;
        } else {
            // This is a duplicate slot, return the direct child pointer
            return slotChild;
        }
    } else {
        // This is a duplicate slot, return the direct child pointer
        return slotChild;
    }
}

void ModelInnerNode::setChild(KeyType key, void* child) {
    bool success = false;

    while (!success) {
        // Predict the slot for this key
        size_t idx = predictSlot(key);

        // Collect slots that need to be locked
        std::vector<size_t> slots_to_lock;
        slots_to_lock.push_back(idx);

        // Find all consecutive duplicate slots to the right
        for (size_t i = idx + 1; i < slots_.size(); i++) {
            KeyType tempKey;
            void* tempChild;
            bool isReal;
            readSlotSeqLock(i, tempKey, tempChild, isReal);

            if (isReal) {
                break;
            }
            slots_to_lock.push_back(i);
        }

        // Try to acquire all locks in reverse order (rightmost first) to avoid deadlock
        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(slots_to_lock.size());

        bool all_locked = true;
        for (auto it = slots_to_lock.rbegin(); it != slots_to_lock.rend(); ++it) {
            std::unique_lock<std::mutex> lock(slots_[*it].lock, std::try_to_lock);
            if (!lock.owns_lock()) {
                // Failed to acquire lock - release all and retry
                all_locked = false;
                break;
            }
            locks.push_back(std::move(lock));
        }

        if (!all_locked) {
            // Release all locks and retry from the beginning
            locks.clear();
            continue;
        }

        // All locks acquired successfully - begin seq lock write phase
        // Increment all versions to odd numbers (in same order as locks were acquired)
        for (auto it = slots_to_lock.rbegin(); it != slots_to_lock.rend(); ++it) {
            slots_[*it].beginWrite();
        }

        // Count leaf nodes in the new subtree
        size_t newChildLeafCount = countLeafNodesInSubtree(child);

        // Update the main slot
        KeyType oldKey;
        void* oldChild;
        bool wasReal;
        readSlotSeqLock(idx, oldKey, oldChild, wasReal);

        if (wasReal) {
            // Slot already has content, adjust leaf count
            size_t oldChildLeafCount = countLeafNodesInSubtree(oldChild);

            if (newChildLeafCount > oldChildLeafCount) {
                leaf_node_count_.fetch_add(newChildLeafCount - oldChildLeafCount, std::memory_order_relaxed);
            } else if (oldChildLeafCount > newChildLeafCount) {
                size_t decrease = std::min(leaf_node_count_.load(std::memory_order_relaxed),
                                           oldChildLeafCount - newChildLeafCount);
                leaf_node_count_.fetch_sub(decrease, std::memory_order_relaxed);
            }
        } else {
            // New slot, add leaf count
            leaf_node_count_.fetch_add(newChildLeafCount, std::memory_order_relaxed);
        }

        // Update slot data
        if (!slots_[idx].isRealChild()) {
            slots_[idx].setRealChild(key, child);
        } else {
            // Safe deletion of old child before replacing
            // void* oldChild = slots_[idx].KeyChildPtr.second;
            // if (oldChild && oldChild != child) {
            //     safeDelete(oldChild);
            // }
            slots_[idx].KeyChildPtr.first = key;
            slots_[idx].KeyChildPtr.second = child;
        }

        // Update duplicate slots with direct child pointers
        for (size_t i = 1; i < slots_to_lock.size(); i++) {
            slots_[slots_to_lock[i]].setDuplicate(child);
        }

        // End seq lock write phase - increment versions to even numbers (reverse order)
        for (size_t i = 0; i < slots_to_lock.size(); i++) {
            slots_[slots_to_lock[i]].endWrite();
        }

        success = true;
        // Locks are automatically released when going out of scope
    }
}

void ModelInnerNode::updateChildWithExternalLock(KeyType key, void* child) {
    bool success = false;

    while (!success) {
        // Predict the slot for this key
        size_t idx = predictSlot(key);

        // Collect additional slots that need to be locked (excluding main slot which should already be locked)
        std::vector<size_t> additional_slots_to_lock;

        // Find all consecutive duplicate slots to the right
        for (size_t i = idx + 1; i < slots_.size(); i++) {
            KeyType tempKey;
            void* tempChild;
            bool isReal;
            readSlotSeqLock(i, tempKey, tempChild, isReal);

            if (isReal) {
                break;
            }
            additional_slots_to_lock.push_back(i);
        }

        // Try to acquire additional locks in reverse order
        std::vector<std::unique_lock<std::mutex>> additional_locks;
        additional_locks.reserve(additional_slots_to_lock.size());

        bool all_additional_locked = true;
        for (auto it = additional_slots_to_lock.rbegin(); it != additional_slots_to_lock.rend(); ++it) {
            std::unique_lock<std::mutex> lock(slots_[*it].lock, std::try_to_lock);
            if (!lock.owns_lock()) {
                // Failed to acquire lock - release all and retry
                all_additional_locked = false;
                break;
            }
            additional_locks.push_back(std::move(lock));
        }

        if (!all_additional_locked) {
            // Release all additional locks and retry from the beginning
            additional_locks.clear();
            continue;
        }

        // All locks acquired successfully - begin seq lock write phase
        // Increment main slot version to odd number (already locked externally)
        slots_[idx].beginWrite();

        // Increment additional slot versions to odd numbers (in reverse order)
        for (auto it = additional_slots_to_lock.rbegin(); it != additional_slots_to_lock.rend(); ++it) {
            slots_[*it].beginWrite();
        }

        // Count leaf nodes in the new subtree
        size_t newChildLeafCount = countLeafNodesInSubtree(child);

        // Update the main slot
        KeyType oldKey;
        void* oldChild;
        bool wasReal;
        readSlotSeqLock(idx, oldKey, oldChild, wasReal);

        if (wasReal) {
            // Slot already has content, adjust leaf count
            size_t oldChildLeafCount = countLeafNodesInSubtree(oldChild);

            if (newChildLeafCount > oldChildLeafCount) {
                leaf_node_count_.fetch_add(newChildLeafCount - oldChildLeafCount, std::memory_order_relaxed);
            } else if (oldChildLeafCount > newChildLeafCount) {
                size_t decrease = std::min(leaf_node_count_.load(std::memory_order_relaxed),
                                           oldChildLeafCount - newChildLeafCount);
                leaf_node_count_.fetch_sub(decrease, std::memory_order_relaxed);
            }
        } else {
            // New slot, add leaf count
            leaf_node_count_.fetch_add(newChildLeafCount, std::memory_order_relaxed);
        }

        // Update slot data
        if (!slots_[idx].isRealChild()) {
            slots_[idx].setRealChild(key, child);
        } else {
            slots_[idx].KeyChildPtr.first = key;
            slots_[idx].KeyChildPtr.second = child;
        }

        // Update duplicate slots with direct child pointers
        for (size_t slotIdx : additional_slots_to_lock) {
            slots_[slotIdx].setDuplicate(child);
        }

        // End seq lock write phase - increment versions to even numbers (forward order)
        for (size_t slotIdx : additional_slots_to_lock) {
            slots_[slotIdx].endWrite();
        }

        // End main slot write phase last
        slots_[idx].endWrite();

        success = true;
        // Additional locks are automatically released when going out of scope
        // Main slot lock remains held by caller
    }
}

void ModelInnerNode::bulkSetChild(KeyType key, void* child) {
    // Predict the slot for this key
    size_t idx = predictSlot(key);

    // Count leaf nodes in the new subtree
    size_t newChildLeafCount = countLeafNodesInSubtree(child);

    // Update slot with the new child pointer
    if (slots_[idx].isRealChild()) {
        // Slot already has content, update it and adjust leaf count
        void* oldChild = slots_[idx].KeyChildPtr.second;
        size_t oldChildLeafCount = countLeafNodesInSubtree(oldChild);

        // Adjust leaf node count based on difference
        if (newChildLeafCount > oldChildLeafCount) {
            leaf_node_count_.fetch_add(newChildLeafCount - oldChildLeafCount, std::memory_order_relaxed);
        } else if (oldChildLeafCount > newChildLeafCount) {
            size_t decrease = std::min(leaf_node_count_.load(std::memory_order_relaxed),
                                       oldChildLeafCount - newChildLeafCount);
            leaf_node_count_.fetch_sub(decrease, std::memory_order_relaxed);
        }

        // Update the key-child pair
        slots_[idx].KeyChildPtr.first = key;
        // Safe deletion of old child before replacing
        // if (oldChild && oldChild != child) {
        //     safeDelete(oldChild);
        // }
        slots_[idx].KeyChildPtr.second = child;
    } else {
        // Initialize new slot with the key-child pair
        slots_[idx].setRealChild(key, child);
        leaf_node_count_.fetch_add(newChildLeafCount, std::memory_order_relaxed);
    }

    // Duplicate this slot to the right for faster lookups
    bulkDuplicateToRight(idx);
}

void ModelInnerNode::bulkLoad(std::vector<std::pair<KeyType, void*>>& leaves,
                              std::vector<int>& slot_counts, double lambda) {
    // Initialize active slots array
    std::vector<int> act_slots(MR_ + 1, 0);

    // Count predicted slots for all leaves
    for (auto & leave : leaves) {
        auto pos = predictSlot(leave.first);
        act_slots[pos]++;
    }

    // Calculate cumulative sums for slot indices
    std::vector<int> sums(slot_counts.size());
    std::partial_sum(slot_counts.begin(), slot_counts.end(), sums.begin());

    // Reset leaf node counts
    leaf_node_count_.store(0, std::memory_order_relaxed);
    initial_leaf_node_count_.store(0, std::memory_order_relaxed);

    // Process each slot with its corresponding leaves
    for (int i = 0; i < slot_counts.size(); ++i) {
        // Skip empty slots
        if (slot_counts[i] == 0) {
            continue;
        }

        // Determine the range of leaves for this slot
        const size_t start = (i == 0) ? 0 : sums[i-1];
        const size_t end = sums[i];
        KeyType slot_key = leaves[start].first;

        if (slot_counts[i] == 1) {
            // Only one leaf assigned to this slot - direct mapping
            bulkSetChild(slot_key, leaves[start].second);
        } else if (slot_counts[i] <= 8) {
            // Small number of leaves - use a search inner node
            auto* search_node = new SearchInnerNode();

            // Extract leaves for this slot
            std::vector<std::pair<KeyType, void*>> sliced_leaves(
                    std::make_move_iterator(leaves.begin() + start),
                    std::make_move_iterator(leaves.begin() + end)
            );

            // Bulk load the search inner node
            search_node->bulk_load(sliced_leaves);
            void* taggedSearch = tagPointer(search_node, NodeType::SearchInner);
            bulkSetChild(slot_key, taggedSearch);
        } else {
            // Large number of leaves - create another model inner node

            // Extract keys for configuration search
            std::vector<KeyType> keys;
            keys.reserve(end - start);
            std::transform(leaves.begin() + start, leaves.begin() + end,
                           std::back_inserter(keys),
                           [](const std::pair<KeyType, void*>& p) { return p.first; });

            // Find optimal configuration for this subset
            ConfigurationSearch cs(keys, lambda);
            auto config = cs.search();

            // Create new model inner node
            auto* model_child_node = new ModelInnerNode(
                    config.best_slope,
                    keys.front(),
                    config.best_mr
            );

            // Extract leaves for this model node
            std::vector<std::pair<KeyType, void*>> sliced_leaves(
                    std::make_move_iterator(leaves.begin() + start),
                    std::make_move_iterator(leaves.begin() + end)
            );

            // Recursively insert into the new model node
            model_child_node->bulkLoad(sliced_leaves, config.best_slot_counts, lambda);
            void* taggedModel = tagPointer(model_child_node, NodeType::ModelInner);
            bulkSetChild(slot_key, taggedModel);
        }
    }

    initial_leaf_node_count_.store(leaf_node_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

void ModelInnerNode::bulkDuplicateToRight(size_t from) {
    // Get the child pointer from the real slot
    void* childPtr = slots_[from].KeyChildPtr.second;

    // Duplicate the child pointer to all slots to its right until a real child is found
    for (size_t i = from + 1; i < slots_.size(); i++) {
        if (slots_[i].isRealChild()) {
            break;
        }
        // Set as duplicate with direct child pointer
        slots_[i].setDuplicate(childPtr);
    }
}

size_t ModelInnerNode::countLeafNodesInSubtree(void* node) const {
    if (!node) return 0;

    // Leaf nodes count as 1
    if (isLeafNode(node)) {
        return 1;
    }

    size_t count = 0;

    // For model inner nodes, use their leaf node count directly
    if (isModelInnerNode(node)) {
        auto* modelNode = taggedCast<ModelInnerNode>(node);
        return modelNode->getLeafNodeCount();
    }
        // For search inner nodes, recursively count leaf nodes
    else if (isSearchInnerNode(node)) {
        auto* searchNode = taggedCast<SearchInnerNode>(node);
        for (const auto& childPair : searchNode->getChildren()) {
            count += countLeafNodesInSubtree(childPair.second);
        }
    }

    return count;
}

void* ModelInnerNode::getChildAtIndex(size_t idx) const {
    // Check index bounds
    if (idx >= slots_.size()) {
        return nullptr;
    }

    KeyType key;
    void* child;
    bool isReal;

    // Read slot with seq lock protocol
    readSlotSeqLock(idx, key, child, isReal);

    // Return child pointer directly (no indirection needed for duplicates)
    return child;
}

std::tuple<std::unique_lock<std::mutex>, void*, size_t> ModelInnerNode::findChildWithLock(KeyType key) {
    size_t idx = predictSlot(key);

    KeyType slotKey;
    void* slotChild;
    bool isReal;

    readSlotSeqLock(idx, slotKey, slotChild, isReal);

    size_t finalSlotIdx = idx;

    if (isReal) {
        // Check if we need to look at the previous slot
        if (slotKey > key && idx > 0) {
            idx--;
            readSlotSeqLock(idx, slotKey, slotChild, isReal);
            finalSlotIdx = idx;
        }

        if (isReal) {
            // Found real child in current slot
            std::unique_lock<std::mutex> lock(slots_[finalSlotIdx].lock);
            return {std::move(lock), slotChild, finalSlotIdx};
        } else {
            // This is a duplicate slot - we need to find the real slot that owns this child
            // Since duplicates now store direct pointers, we need to search for the real slot
            void* targetChild = slotChild;
            for (size_t i = 0; i < slots_.size(); i++) {
                if (slots_[i].isRealChild() && slots_[i].KeyChildPtr.second == targetChild) {
                    finalSlotIdx = i;
                    break;
                }
            }
        }
    } else {
        // This is a duplicate slot - we need to find the real slot that owns this child
        void* targetChild = slotChild;
        for (size_t i = 0; i < slots_.size(); i++) {
            if (slots_[i].isRealChild() && slots_[i].KeyChildPtr.second == targetChild) {
                finalSlotIdx = i;
                slotChild = targetChild;
                break;
            }
        }
    }

    std::unique_lock<std::mutex> lock(slots_[finalSlotIdx].lock);
    return {std::move(lock), slotChild, finalSlotIdx};
}

bool ModelInnerNode::areAllSlotsUnlocked() const {
    for (const auto& slot : slots_) {
        if (slot.lock.try_lock()) {
            slot.lock.unlock();
        } else {
            return false;  // Slot is locked
        }
    }
    return true;
}

bool ModelInnerNode::isSubtreeUnlocked() const {
    // First, check if all slots in this node are unlocked
    if (!areAllSlotsUnlocked()) {
        return false;
    }

    // Then check all children recursively
    for (size_t i = 0; i < slots_.size(); i++) {
        KeyType key;
        void* child;
        bool isReal;

        readSlotSeqLock(i, key, child, isReal);

        if (isReal && child != nullptr) {
            if (isModelInnerNode(child)) {
                auto* modelChild = taggedCast<ModelInnerNode>(child);
                if (!modelChild->isSubtreeUnlocked()) {
                    return false;
                }
            } else if (isSearchInnerNode(child)) {
                auto* searchChild = taggedCast<SearchInnerNode>(child);
                if (!searchChild->areAllSlotsUnlocked()) {
                    return false;
                }
            }
            // Leaf nodes don't have locks, so they're always "unlocked"
        }
    }

    return true;
}

void ModelInnerNode::increaseLeafNodeCount() {
    leaf_node_count_.fetch_add(1, std::memory_order_relaxed);
}

size_t ModelInnerNode::getLeafNodeCount() const {
    return leaf_node_count_.load(std::memory_order_relaxed);
}

bool ModelInnerNode::shouldRebuild() const {
    size_t curr = leaf_node_count_.load(std::memory_order_relaxed);
    size_t init = initial_leaf_node_count_.load(std::memory_order_relaxed);
    return init > 0 && curr >= 10 * init;
}

void ModelInnerNode::deleteChildNode(void* childPtr) {
    if (childPtr == nullptr) return;
    
    if (isLeafNode(childPtr)) {
        delete taggedCast<LeafNode>(childPtr);
    } else if (isModelInnerNode(childPtr)) {
        delete taggedCast<ModelInnerNode>(childPtr);
    } else if (isSearchInnerNode(childPtr)) {
        delete taggedCast<SearchInnerNode>(childPtr);
    }
}

std::set<void*> ModelInnerNode::getAllUniqueChildren() const {
    std::set<void*> uniqueChildren;
    
    for (const auto& slot : slots_) {
        void* childPtr = nullptr;
        
        if (slot.isRealChild()) {
            childPtr = slot.KeyChildPtr.second;
        } else {
            childPtr = slot.getChildPtr();
        }
        
        if (childPtr != nullptr) {
            uniqueChildren.insert(childPtr);
        }
    }
    
    return uniqueChildren;
}