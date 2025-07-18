#include "../include/search_inner_node.h"
#include "../include/leaf_node.h"
#include "../include/model_inner_node.h"
#include "../include/epoch_manager.h"

SearchInnerNode::~SearchInnerNode() {
    for (auto& entry : children_) {
        if (entry.childPtr != nullptr) {
            if (isLeafNode(entry.childPtr)) {
                delete taggedCast<LeafNode>(entry.childPtr);
            } else if (isModelInnerNode(entry.childPtr)) {
                delete taggedCast<ModelInnerNode>(entry.childPtr);
            } else if (isSearchInnerNode(entry.childPtr)) {
                delete taggedCast<SearchInnerNode>(entry.childPtr);
            }
        }
    }
    children_.clear();
}

void SearchInnerNode::addChild(KeyType boundaryKey, void* child) {
    std::lock_guard<std::mutex> guard(structural_lock_);

    incrementVersion();

    // Find the insertion position using binary search
    auto it = std::lower_bound(children_.begin(), children_.end(), boundaryKey,
                               [](const ChildEntry& entry, const KeyType& key) {
                                   return entry.boundaryKey < key;
                               });

    // If an element with the same boundaryKey exists, replace its child
    if (it != children_.end() && it->boundaryKey == boundaryKey) {
        // Simply update the existing entry
        it->childPtr = child;
        // delete old node
    } else {
        // Insert the new element at the found position
        children_.emplace(it, boundaryKey, child);
    }

    incrementVersion();
}

void* SearchInnerNode::findChild(KeyType key) const {
    return readWithVersionCheck([this, key]() -> void* {
        if (children_.empty()) {
            return nullptr;
        }

        // Binary search to find the last child with key <= search key
        auto it = std::upper_bound(children_.begin(), children_.end(), key,
                                   [](const KeyType& key, const ChildEntry& entry) {
                                       return key < entry.boundaryKey;
                                   });

        if (it != children_.begin()) {
            return std::prev(it)->childPtr;
        }

        return nullptr;
    });
}

void SearchInnerNode::bulk_load(const std::vector<std::pair<KeyType, void*>>& bulkChildren) {
    children_.reserve(bulkChildren.size());

    for (const auto& [key, ptr] : bulkChildren) {
        children_.emplace_back(key, ptr);
    }
}

std::vector<std::pair<KeyType, void*>> SearchInnerNode::getChildren() const {
    return readWithVersionCheck([this]() -> std::vector<std::pair<KeyType, void*>> {
        std::vector<std::pair<KeyType, void*>> result;
        result.reserve(children_.size());

        for (const auto& entry : children_) {
            result.emplace_back(entry.boundaryKey, entry.childPtr);
        }
        return result;
    });
}


int SearchInnerNode::findChildIndex(KeyType key) const {
    return readWithVersionCheck([this, key]() -> int {
        if (children_.empty()) {
            return -1;
        }

        // Upper bound returns the first element greater than key
        auto it = std::upper_bound(children_.begin(), children_.end(), key,
                                   [](const KeyType& key, const ChildEntry& entry) {
                                       return key < entry.boundaryKey;
                                   });

        if (it != children_.begin()) {
            return std::distance(children_.begin(), std::prev(it));
        }

        return -1;
    });
}

void* SearchInnerNode::getChildAtIndex(int idx) const {
    return readWithVersionCheck([this, idx]() -> void* {
        if (idx < 0 || idx >= static_cast<int>(children_.size())) {
            return nullptr;
        }
        return children_[idx].childPtr;
    });
}

std::pair<std::mutex*, void*> SearchInnerNode::getSlotLockAndChild(int idx) {
    // For lock access, we need to hold structural lock to ensure stability
    std::lock_guard<std::mutex> guard(structural_lock_);

    if (idx < 0 || idx >= static_cast<int>(children_.size())) {
        return {nullptr, nullptr};
    }

    // Return both lock pointer and child pointer atomically
    return {children_[idx].lock.get(), children_[idx].childPtr};
}

uint64_t SearchInnerNode::getVersion() const {
    return version_.load(std::memory_order_acquire);
}

void SearchInnerNode::incrementVersion() {
    version_.fetch_add(1, std::memory_order_release);
}

bool SearchInnerNode::areAllSlotsUnlocked() const {
    return readWithVersionCheck([this]() -> bool {
        for (const auto& entry : children_) {
            if (entry.lock && entry.lock->try_lock()) {
                entry.lock->unlock();
            } else {
                return false;  // Slot is locked
            }
        }
        return true;
    });
}