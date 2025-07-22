#include "../include/hyper_index.h"
#include "../include/configuration_search.h"
#include "../include/epoch_manager.h"
#include <algorithm>

Hyper::~Hyper() {
    // Clean up the root node and its children using safe deletion
    void* currentRoot = root_.load(std::memory_order_acquire);
    if (currentRoot) {
        if (isModelInnerNode(currentRoot)) {
            delete taggedCast<ModelInnerNode>(currentRoot);
        } else if (isSearchInnerNode(currentRoot)) {
            delete taggedCast<SearchInnerNode>(currentRoot);
        } else if (isLeafNode(currentRoot)) {
            delete taggedCast<LeafNode>(currentRoot);
        }
        root_.store(nullptr, std::memory_order_release);
    }
    
    // Force final memory reclamation
    EpochManager::get().forceReclamation();
}

void Hyper::bulkLoad(const std::vector<std::pair<KeyType, ValueType>>& data) {
    if (data.empty()) return;

    // Define the key value boundary between the two partitions.
    constexpr KeyType cutValue = 1ULL << 63;

    // Check if the sentinel keys are already in the data.
    // Since the data is sorted, data.front() is the smallest key.
    bool needLowSentinel = (data.front().first != 0);
    bool needHighSentinel = !std::any_of(data.begin(), data.end(),
                                         [](const std::pair<KeyType, ValueType>& kv) {
                                             return kv.first == (1ULL << 63);
                                         });

    // Partition the data into low and high parts.
    auto partitionIt = std::lower_bound(data.begin(), data.end(), cutValue,
                                        [](const std::pair<KeyType, ValueType>& kv, KeyType value) {
                                            return kv.first < value;
                                        });

    // Prepare containers for low and high key ranges.
    std::vector<std::pair<KeyType, ValueType>> lowData;
    std::vector<std::pair<KeyType, ValueType>> highData;
    lowData.reserve(std::distance(data.begin(), partitionIt) + (needLowSentinel ? 1 : 0));
    highData.reserve(std::distance(partitionIt, data.end()) + (needHighSentinel ? 1 : 0));

    // Add low-sentinel if needed.
    if (needLowSentinel)
        lowData.emplace_back(0, std::numeric_limits<ValueType>::max());

    // Insert the low partition keys.
    lowData.insert(lowData.end(),
                   std::make_move_iterator(const_cast<std::pair<KeyType, ValueType>*>(&data[0])),
                   std::make_move_iterator(const_cast<std::pair<KeyType, ValueType>*>(&*partitionIt)));

    // Add high-sentinel if needed.
    if (needHighSentinel)
        highData.emplace_back(cutValue, std::numeric_limits<ValueType>::max());

    // Insert the high partition keys.
    highData.insert(highData.end(),
                    std::make_move_iterator(const_cast<std::pair<KeyType, ValueType>*>(&*partitionIt)),
                    std::make_move_iterator(const_cast<std::pair<KeyType, ValueType>*>(&data[0] + data.size())));

    // Build leaf nodes for each partition.
    auto lowLeaves = buildLeavesForPartition(std::move(lowData));
    auto highLeaves = buildLeavesForPartition(std::move(highData));

    // Combine all leaf nodes into a single vector.
    std::vector<std::pair<KeyType, void*>> leaves;
    leaves.reserve(lowLeaves.size() + highLeaves.size());
    leaves.insert(leaves.end(),
                  std::make_move_iterator(lowLeaves.begin()),
                  std::make_move_iterator(lowLeaves.end()));
    leaves.insert(leaves.end(),
                  std::make_move_iterator(highLeaves.begin()),
                  std::make_move_iterator(highLeaves.end()));

    // Extract boundary keys from the leaf nodes.
    std::vector<KeyType> boundary_keys;
    boundary_keys.resize(leaves.size());
    std::transform(leaves.begin(), leaves.end(), boundary_keys.begin(),
                   [](const std::pair<KeyType, void*>& p) { return p.first; });

    // Find optimal configuration for the root node.
    ConfigurationSearch cs(boundary_keys, lambda_);
    auto config = cs.search();

    // Create the root model node with optimal parameters.
    auto root = new ModelInnerNode(
            config.best_slope,
            boundary_keys.front(),
            config.best_mr
    );

    // Insert all leaf nodes into the root.
    root->bulkLoad(leaves, config.best_slot_counts, lambda_);

    // Store the tagged root pointer.
    root_ = tagPointer(root, NodeType::ModelInner);
}


std::optional<ValueType> Hyper::find(KeyType key) const {
    //EpochManager::Guard guard; // Protect the entire find operation
    
    void* cur = root_.load(std::memory_order_acquire);

    // Traverse the tree to find the leaf node containing the key
    while (!isLeafNode(cur)) {
        if (isModelInnerNode(cur)) {
            auto* modelNode = taggedCast<ModelInnerNode>(cur);
            void* nextNode = modelNode->findChild(key);
            __builtin_prefetch(nextNode, 0, 3);
            cur = nextNode;
        } else if (isSearchInnerNode(cur)) {
            auto* searchNode = taggedCast<SearchInnerNode>(cur);
            void* nextNode = searchNode->findChild(key);
            __builtin_prefetch(nextNode, 0, 3);
            cur = nextNode;
        } else {
            return std::nullopt;  // Unknown node type
        }
    }

    // Search within the leaf node
    auto* leaf = taggedCast<LeafNode>(cur);
    auto result = leaf->find(key);

    // Special handling for sentinel values
    if (*result == std::numeric_limits<ValueType>::max()) {
        if ((key == 0) | (key == (1ULL << 63))) {
            return std::nullopt;
        } else if (key != std::numeric_limits<KeyType>::max()) {
            find(key);
        }
    }
    return result;
}

void Hyper::insert(KeyType key, ValueType value) {
    EpochManager::Guard guard; // Protect the entire insert operation
    
    // Initialize the index with a single leaf node if empty
    void* currentRoot = root_.load(std::memory_order_acquire);
    if (currentRoot == nullptr) {
        // Only lock for initial root creation
        std::lock_guard<std::mutex> rootGuard(root_update_lock_);
        
        // Double-check after acquiring lock
        currentRoot = root_.load(std::memory_order_acquire);
        if (currentRoot != nullptr) {
            // Another thread already initialized, proceed with normal insert
            for (int attempt = 0; attempt < MAX_INSERT_RETRIES; ++attempt) {
                InsertResult result = insertAttempt(key, value);
                if (result != InsertResult::RetryFromRoot) {
                    if (attempt > 0) {
                        std::cout << "Insert succeeded after " << (attempt + 1) << " attempts for key " << key << std::endl;
                    }
                    return;
                }
                
                if (attempt >= 3) {
                    std::cout << "Insert retry " << (attempt + 1) << " for key " << key 
                              << " (outdated node encountered)" << std::endl;
                }
                
                if (attempt < MAX_INSERT_RETRIES - 1) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1 << std::min(attempt, 10)));
                }
            }
            
            std::cerr << "CRITICAL ERROR: Insert operation failed after " << MAX_INSERT_RETRIES 
                      << " retries for key " << key << std::endl;
            throw std::runtime_error("Insert operation exceeded maximum retries");
        }
        
        // Create initial root structure
        LeafNode* lowerLeaf;
        LeafNode* upperLeaf;
        if (key < (1ULL << 63)) {
            lowerLeaf = new LeafNode(1.0, 0, key);
            lowerLeaf->insert(key, value, delta_);
            lowerLeaf->insert(0, std::numeric_limits<ValueType>::max(), delta_);
            upperLeaf = new LeafNode(1.0, (1ULL << 63), (1ULL << 63));
            upperLeaf->insert(1ULL << 63, std::numeric_limits<ValueType>::max(), delta_);
        } else {
            lowerLeaf = new LeafNode(1.0, 0, 0);
            lowerLeaf->insert(0, std::numeric_limits<ValueType>::max(), delta_);
            upperLeaf = new LeafNode(1.0, (1ULL << 63), key);
            upperLeaf->insert(1ULL << 63, std::numeric_limits<ValueType>::max(), delta_);
            upperLeaf->insert(key, value, delta_);
        }

        ModelInnerNode* rootInner = new ModelInnerNode(1.0, 0, 1);
        rootInner->setChild(0, tagPointer(lowerLeaf, NodeType::Leaf));
        rootInner->setChild(1ULL << 63, tagPointer(upperLeaf, NodeType::Leaf));

        root_.store(tagPointer(rootInner, NodeType::ModelInner), std::memory_order_release);
        return;
    }

    // Normal insert without root lock
    for (int attempt = 0; attempt < MAX_INSERT_RETRIES; ++attempt) {
        InsertResult result = insertAttempt(key, value);
        if (result != InsertResult::RetryFromRoot) {
            if (attempt > 0) {
                std::cout << "Insert succeeded after " << (attempt + 1) << " attempts for key " << key << std::endl;
            }
            return;
        }
        
        if (attempt >= 3) {
            std::cout << "Insert retry " << (attempt + 1) << " for key " << key 
                      << " (outdated node encountered)" << std::endl;
        }
        
        if (attempt < MAX_INSERT_RETRIES - 1) {
            std::this_thread::sleep_for(std::chrono::microseconds(1 << std::min(attempt, 10)));
        }
    }
    
    std::cerr << "CRITICAL ERROR: Insert operation failed after " << MAX_INSERT_RETRIES 
              << " retries for key " << key << std::endl;
    throw std::runtime_error("Insert operation exceeded maximum retries");
}

InsertResult Hyper::insertAttempt(KeyType key, ValueType value) {
    // Load root without locking - this is the key change!
    void* cur = root_.load(std::memory_order_acquire);
    void* rootSnapshot = cur;  // Keep snapshot for potential root update
    void* parent = nullptr;
    size_t parentSlotIdx = 0;

    // Traverse to find the target leaf node
    while (!isLeafNode(cur)) {
        parent = cur;

        if (isModelInnerNode(cur)) {
            auto* modelNode = taggedCast<ModelInnerNode>(cur);
            auto [slotLock, nextNode, slotIdx] = modelNode->findChildWithLock(key);
            parentSlotIdx = slotIdx;

            // Release lock immediately - we only need it for split operations
            slotLock.unlock();

            __builtin_prefetch(nextNode, 0, 3);
            cur = nextNode;

        } else if (isSearchInnerNode(cur)) {
            auto* searchNode = taggedCast<SearchInnerNode>(cur);
            parentSlotIdx = searchNode->findChildIndex(key);

            auto [slotMutexPtr, nextNode] = searchNode->getSlotLockAndChild(parentSlotIdx);

            {
                std::lock_guard<std::mutex> slot_guard(*slotMutexPtr);
                __builtin_prefetch(nextNode, 0, 3);
                cur = nextNode;
            }
        } else {
            break;
        }
    }

    // Insert into the leaf node
    LeafNode* leaf = taggedCast<LeafNode>(cur);
    InsertReturn insertReturn = leaf->insert(key, value, delta_);

    // Handle the different return cases
    switch (insertReturn.result) {
        case InsertResult::Success:
            return InsertResult::Success;
            
        case InsertResult::SuccessWithSplit:
            // Split occurred - check if we need root update
            if (insertReturn.splitDescriptors.has_value()) {
                // Check if we need to update root
                if (parent == nullptr || needsRootUpdate(rootSnapshot, (*insertReturn.splitDescriptors)[0])) {
                    // We need to update the root using RCU
                    return handleRootUpdate(rootSnapshot, *insertReturn.splitDescriptors);
                } else {
                    // Normal split, insert descriptors
                    std::unique_lock<std::mutex> parentLock;
                    insertLeafDescriptors(*insertReturn.splitDescriptors, parentLock);
                }
            }
            return InsertResult::Success;
            
        case InsertResult::RetryFromRoot:
            return InsertResult::RetryFromRoot;
    }

    return InsertResult::Success;
}

void Hyper::insertLeafDescriptors(const std::vector<std::pair<KeyType, void*>>& leafDescs,
                                  std::unique_lock<std::mutex>& parentLock) {
    // Track model nodes that might need rebuilding
    std::set<std::pair<ModelInnerNode*, void*>> modelNodesWithParent;
    std::set<std::pair<SearchInnerNode*, void*>> searchNodesWithParent;

    for (const auto& leafDesc : leafDescs) {
        void* cur = root_;
        void* parentNode = nullptr;

        // Tree traversal to find insertion point
        while (!isLeafNode(cur)) {
            if (isModelInnerNode(cur)) {
                auto* mNode = taggedCast<ModelInnerNode>(cur);
                modelNodesWithParent.insert({mNode, parentNode});

                size_t slotIdx = mNode->getSlotIndex(leafDesc.first);

                // If this is a duplicate slot, we can directly update
                if (mNode->isDuplicateSlot(slotIdx)) {
                    mNode->setChild(leafDesc.first, leafDesc.second);
                    break;
                } else {
                    void* childPtr = mNode->getChildAtIndex(slotIdx);
                    if (childPtr != nullptr && isLeafNode(childPtr)) {
                        LeafNode* oldLeaf = taggedCast<LeafNode>(childPtr);
                        // Check if it's the last leaf descriptor
                        if (&leafDesc == &leafDescs.back()
                            && oldLeaf->getMinKey() == taggedCast<LeafNode>(leafDesc.second)->getMinKey()) {
                                mNode->updateChildWithExternalLock(leafDesc.first, leafDesc.second);
                                // delete old leaf
                        } else {
                            // Create search inner node
                            auto* newSearchNode = new SearchInnerNode();
                            auto* oldLeaf = taggedCast<LeafNode>(childPtr);
                            newSearchNode->addChild(oldLeaf->getMinKey(), childPtr);
                            newSearchNode->addChild(leafDesc.first, leafDesc.second);
                            mNode->setChild(std::min(oldLeaf->getMinKey(), leafDesc.first),
                                            tagPointer(newSearchNode, NodeType::SearchInner));
                        }
                        break;
                    } else if (childPtr != nullptr && isModelInnerNode(childPtr)) {
                        auto* childModelNode = taggedCast<ModelInnerNode>(childPtr);

                        if (leafDesc.first < childModelNode->getMinKey()) {
                            // Special case: Need to rebuild subtree

                            // Lock the slot to ensure consistent reads during collectAllData
                            std::mutex& slotLock = mNode->getSlotLock(slotIdx);
                            std::lock_guard<std::mutex> slotGuard(slotLock);

                            std::vector<std::pair<KeyType, ValueType>> childData;
                            collectAllData(leafDesc.second, childData);
                            collectAllData(childPtr, childData);

                            // Build new leaves from combined data
                            std::vector<std::pair<KeyType, void*>> newLeaves = buildLeaves(std::move(childData));

                            // Extract boundary keys for configuration search
                            std::vector<KeyType> boundaryKeys;
                            boundaryKeys.resize(newLeaves.size());
                            std::transform(newLeaves.begin(), newLeaves.end(), boundaryKeys.begin(),
                                           [](const std::pair<KeyType, void*>& p) {return p.first;});

                            // Find optimal configuration
                            ConfigurationSearch cs(boundaryKeys, lambda_);
                            auto config = cs.search();

                            // Create new model node
                            auto* newModelNode = new ModelInnerNode(
                                    config.best_slope,
                                    boundaryKeys.front(),
                                    config.best_mr
                            );

                            // Insert all leaves into the new model
                            newModelNode->bulkLoad(newLeaves, config.best_slot_counts, lambda_);
                            void* taggedNewModel = tagPointer(newModelNode, NodeType::ModelInner);

                            // Update parent to point to the new model
                            mNode->updateChildWithExternalLock(boundaryKeys.front(), taggedNewModel);

                            // Clean up old model node

                            break;
                        }
                    }
                }

                // Continue traversal
                mNode->increaseLeafNodeCount();
                parentNode = tagPointer(mNode, NodeType::ModelInner);
                cur = mNode->getChildAtIndex(slotIdx);
            } else if (isSearchInnerNode(cur)) {
                auto* sNode = taggedCast<SearchInnerNode>(cur);
                void* candidate = sNode->findChild(leafDesc.first);

                // Handle search inner node cases
                if (candidate == nullptr) {
                    // No child found - direct insertion
                    sNode->addChild(leafDesc.first, leafDesc.second);

                    // Update parent pointers
                    if (parentNode && isModelInnerNode(parentNode)) {
                        taggedCast<ModelInnerNode>(parentNode)->setChild(leafDesc.first, tagPointer(sNode, NodeType::SearchInner));
                    } else if (parentNode && isSearchInnerNode(parentNode)) {
                        taggedCast<SearchInnerNode>(parentNode)->addChild(leafDesc.first, tagPointer(sNode, NodeType::SearchInner));
                    }

                    // Convert to model node if needed
                    if (sNode->getChildCount() > 8) {
                        searchNodesWithParent.insert({sNode, parentNode});
                    }
                    break;
                } else if (!isLeafNode(candidate)) {
                    // Continue traversal
                    parentNode = tagPointer(sNode, NodeType::SearchInner);
                    cur = candidate;
                    continue;
                } else {
                    // Found a leaf, add new child
                    sNode->addChild(leafDesc.first, leafDesc.second);

                    // Convert to model if needed
                    if (sNode->getChildCount() > 8) {
                        searchNodesWithParent.insert({sNode, parentNode});
                    }
                    break;
                }
            }
        }
    }

    // Release the parent lock before doing conversions and rebuilds
    // This ensures no locks are held during collectAllData calls
    if (parentLock.owns_lock()) {
        parentLock.unlock();
    }

    for (const auto& [searchNode, parent] : searchNodesWithParent) {
        convertSearchNodeToModelNode(searchNode, parent);
    }

    // Check if any model nodes should be rebuilt
    for (const auto& [modelNode, parent] : modelNodesWithParent) {
        if (modelNode->shouldRebuild()) {
            rebuildModelNode(modelNode, parent);
            break;
        }
    }


}


std::vector<std::pair<KeyType, ValueType>> Hyper::rangeQuery(KeyType left, KeyType right) const {
    EpochManager::Guard guard; // Protect the entire range query operation
    
    std::vector<std::pair<KeyType, ValueType>> result;

    // Early exit for invalid range or empty index
    if (left > right || root_ == nullptr) {
        return result;
    }

    // Structure to store the path from root to leaf
    struct PathEntry {
        void* node;
        size_t index;       // Index within the node
        bool isModelNode;   // Type flag
    };
    std::vector<PathEntry> path;

    // Phase 1: Find the leaf node containing the left boundary
    void* current = root_;
    while (!isLeafNode(current)) {
        if (isModelInnerNode(current)) {
            auto* modelNode = taggedCast<ModelInnerNode>(current);
            size_t idx = modelNode->getSlotIndex(left);
            path.push_back({current, idx, true});
            current = modelNode->findChild(left);
        } else if (isSearchInnerNode(current)) {
            auto* searchNode = taggedCast<SearchInnerNode>(current);
            int idx = searchNode->findChildIndex(left);
            path.push_back({current, static_cast<size_t>(idx), false});
            current = searchNode->findChild(left);
        }

        if (current == nullptr) {
            return result;
        }
    }

    // Phase 2: Single loop to process leaves until we hit the right boundary
    bool rightBoundaryHit = false;
    bool isFirstLeaf = true;

    while (current != nullptr && isLeafNode(current) && !rightBoundaryHit) {
        // Process current leaf efficiently
        auto* leaf = taggedCast<LeafNode>(current);

        // For first leaf, compute starting slot position based on left boundary
        // For subsequent leaves, start from the beginning
        size_t startSlot = 0;
        if (isFirstLeaf) {
            startSlot = leaf->getSlotIndex(left);
            isFirstLeaf = false;
        }

        // Only scan from the computed start position
        for (size_t i = startSlot; i < leaf->getSlots().size(); i++) {
            const auto& slot = leaf->getSlots()[i];

            // Process single key-value pair
            if (slot.isKV()) {
                KeyType key = leaf->decodeKey(slot.data.kv.key);
                if (key >= left && key <= right) {
                    result.emplace_back(key, slot.data.kv.value);
                }
                if (key > right) {
                    rightBoundaryHit = true;
                    break;
                }
            }
                // Process overflow buffer
            else if (slot.isPointer() && slot.data.overflowPtr) {
                const auto& buffer_data = slot.data.overflowPtr.load(std::memory_order_acquire)->data();

                // Binary search for the lower bound in the overflow buffer
                auto startIter = buffer_data.begin();
                if (isFirstLeaf) {
                    startIter = std::lower_bound(buffer_data.begin(), buffer_data.end(), left,
                                                 [](const auto& pair, const KeyType key) {
                                                     return pair.first < key;
                                                 });
                }

                for (auto it = startIter; it != buffer_data.end(); ++it) {
                    if (it->first <= right) {
                        result.push_back(*it);
                    } else {
                        rightBoundaryHit = true;
                        break;
                    }
                }

                if (rightBoundaryHit) break;
            }
        }

        if (rightBoundaryHit) break;

        // Find next leaf node
        void* next_leaf = nullptr;

        // Backtrack up the path to find the next sibling
        while (!path.empty() && next_leaf == nullptr) {
            PathEntry entry = path.back();
            path.pop_back();

            if (entry.isModelNode) {
                auto* modelNode = taggedCast<ModelInnerNode>(entry.node);

                // Try the next slot
                size_t nextIdx = entry.index + 1;
                if (nextIdx < modelNode->getNumSlots()) {
                    void* child = modelNode->getChildAtIndex(nextIdx);
                    if (child) {
                        // Add the new path entry
                        path.push_back({entry.node, nextIdx, true});

                        // Descend to the leftmost leaf
                        while (!isLeafNode(child)) {
                            if (isModelInnerNode(child)) {
                                auto* mNode = taggedCast<ModelInnerNode>(child);
                                path.push_back({child, 0, true});
                                child = mNode->getChildAtIndex(0);
                            } else if (isSearchInnerNode(child)) {
                                auto* sNode = taggedCast<SearchInnerNode>(child);
                                path.push_back({child, 0, false});
                                child = sNode->getChildAtIndex(0);
                            }
                        }
                        next_leaf = child;
                    }
                }
            }
            else {  // Search inner node
                auto* searchNode = taggedCast<SearchInnerNode>(entry.node);

                // Try the next child
                int nextIdx = static_cast<int>(entry.index) + 1;
                if (nextIdx < searchNode->getChildCount()) {
                    void* child = searchNode->getChildAtIndex(nextIdx);

                    // Add the new path entry
                    path.push_back({entry.node, static_cast<size_t>(nextIdx), false});

                    // Descend to the leftmost leaf
                    while (!isLeafNode(child)) {
                        if (isModelInnerNode(child)) {
                            auto* mNode = taggedCast<ModelInnerNode>(child);
                            path.push_back({child, 0, true});
                            child = mNode->getChildAtIndex(0);
                        } else if (isSearchInnerNode(child)) {
                            auto* sNode = taggedCast<SearchInnerNode>(child);
                            path.push_back({child, 0, false});
                            child = sNode->getChildAtIndex(0);
                        }
                    }
                    next_leaf = child;
                }
            }
        }

        // Early termination check - if next leaf's minimum key exceeds our range
        if (next_leaf) {
            auto* nextLeafNode = taggedCast<LeafNode>(next_leaf);
            if (nextLeafNode->getMinKey() > right) {
                break;
            }
        }

        current = next_leaf;
    }

    return result;
}


std::vector<Hyper::PLASegment> Hyper::buildPlaSegments(
        const std::vector<std::pair<KeyType, ValueType>>& data) {
    std::vector<PLASegment> segments;
    KeyType maxKey = data.back().first;

    // Use PLA segmentation to create linear segments
    hyperpgm::internal::make_segmentation_par(
            data.size(),
            static_cast<size_t>(delta_),
            [&](auto i) { return data[i].first; },
            [&](const auto& cs) {
                PLASegment seg{};
                seg.min_key = cs.get_first_x();
                seg.max_key = cs.get_last_x();
                if (cs.get_last_x() >= maxKey) {
                    seg.max_key = data.back().first;
                }
                auto [slope, intercept] = cs.get_floating_point_segment(seg.min_key);
                seg.slope = slope;

                // Determine boundaries using lower_bound on data
                auto start_it = std::lower_bound(data.begin(), data.end(), seg.min_key,
                                                 [](const auto &pair, KeyType key) { return pair.first < key; });
                seg.start_idx = std::distance(data.begin(), start_it);
                auto end_it = std::lower_bound(data.begin(), data.end(), seg.max_key,
                                               [](const auto &pair, KeyType key) { return pair.first < key; });
                seg.end_idx = std::distance(data.begin(), end_it);
                segments.push_back(seg);
            }
    );

    // Remove invalid last segment
    if (!segments.empty() && segments.back().start_idx >= data.size()) {
        segments.pop_back();
    }

    return segments;
}

std::vector<std::pair<KeyType, void*>> Hyper::buildLeavesForPartition(
        const std::vector<std::pair<KeyType, ValueType>>&& data) {
    // Generate PLA segments for the data
    const std::vector<PLASegment> segments = buildPlaSegments(data);

    // Create leaf nodes based on segments
    std::vector<std::pair<KeyType, void*>> keyLeafPtrs;
    keyLeafPtrs.reserve(segments.size());

    for (const auto& seg : segments) {
        // Create a leaf node for this segment
        auto* leaf = new LeafNode(seg.slope, seg.min_key, data[seg.end_idx].first);

        // Extract data for this segment
        std::vector<std::pair<KeyType, ValueType>> segData(
                std::make_move_iterator(const_cast<std::pair<KeyType, ValueType>*>(&data[seg.start_idx])),
                std::make_move_iterator(const_cast<std::pair<KeyType, ValueType>*>(&data[seg.end_idx] + 1))
        );

        // Bulk load the leaf node
        leaf->bulkLoad(std::move(segData));

        // Tag and store the leaf node
        void* taggedLeaf = tagPointer(leaf, NodeType::Leaf);
        keyLeafPtrs.emplace_back(seg.min_key, taggedLeaf);
    }

    return keyLeafPtrs;
}

std::vector<std::pair<KeyType, void*>> Hyper::buildLeaves(
        const std::vector<std::pair<KeyType, ValueType>>&& data) {
    constexpr KeyType cutValue = 1ULL << 63;
    if (data.empty()) return {};

    // Check if data spans both low and high ranges
    bool hasLow  = data.front().first < cutValue;
    bool hasHigh = data.back().first >= cutValue;
    if (hasLow && hasHigh) {
        // Split data at cutValue
        auto partitionIt = std::lower_bound(data.begin(), data.end(), cutValue,
                                            [](const std::pair<KeyType, ValueType>& kv, KeyType value) {
                                                return kv.first < value;
                                            });

        // Create temporary containers for the split data
        std::vector<std::pair<KeyType, ValueType>> lowData(data.begin(), partitionIt);
        std::vector<std::pair<KeyType, ValueType>> highData(partitionIt, data.end());

        // Build leaves for each partition
        auto leavesLow = buildLeavesForPartition(std::move(lowData));
        auto leavesHigh = buildLeavesForPartition(std::move(highData));

        // Combine the results
        leavesLow.insert(leavesLow.end(),
                         std::make_move_iterator(leavesHigh.begin()),
                         std::make_move_iterator(leavesHigh.end()));
        return leavesLow;
    } else {
        // No need to split, process as a single partition
        return buildLeavesForPartition(std::move(const_cast<std::vector<std::pair<KeyType, ValueType>>&>(data)));
    }
}

void Hyper::rebuildModelNode(ModelInnerNode* modelNode, void* parentNode) {
    std::unique_lock<std::mutex> parentLock;

    if (parentNode == nullptr) {
        // This is the root node - use RCU update instead of locking
        void* oldRoot = tagPointer(modelNode, NodeType::ModelInner);
        
        // Collect all data from the model node subtree
        std::vector<std::pair<KeyType, ValueType>> allData;
        collectAllData(oldRoot, allData);

        // Build new leaves from collected data
        std::vector<std::pair<KeyType, void*>> newLeaves = buildLeaves(std::move(allData));

        // Extract boundary keys for configuration search
        std::vector<KeyType> newBoundaryKeys;
        newBoundaryKeys.resize(newLeaves.size());
        std::transform(newLeaves.begin(), newLeaves.end(), newBoundaryKeys.begin(),
                       [](const std::pair<KeyType, void*>& p) { return p.first; });

        // Find optimal configuration
        ConfigurationSearch cs(newBoundaryKeys, lambda_);
        auto config = cs.search();

        // Create new model node with optimal parameters
        auto* newModelNode = new ModelInnerNode(
                config.best_slope,
                newBoundaryKeys.front(),
                config.best_mr
        );

        // Insert all leaves into the new model
        newModelNode->bulkLoad(newLeaves, config.best_slot_counts, lambda_);
        void* taggedNewModelNode = tagPointer(newModelNode, NodeType::ModelInner);

        // Try RCU update
        if (!updateRootRCU(oldRoot, taggedNewModelNode)) {
            // Root changed during rebuild, clean up and let caller retry
            safeDelete(newModelNode);
            return;
        }
        
        // Successfully updated root via RCU
        return;
        
    } else if (isModelInnerNode(parentNode)) {
        // Parent is a model inner node - find and lock the appropriate slot
        auto* modelParent = taggedCast<ModelInnerNode>(parentNode);

        // Find which slot contains this model node
        size_t slotIdx = 0;
        bool found = false;
        void* taggedModelNode = tagPointer(modelNode, NodeType::ModelInner);

        for (size_t i = 0; i < modelParent->getNumSlots(); i++) {
            void* child = modelParent->getChildAtIndex(i);
            if (child == taggedModelNode) {
                slotIdx = i;
                found = true;
                break;
            }
        }

        if (found) {
            parentLock = std::unique_lock<std::mutex>(modelParent->getSlotLock(slotIdx));
        }
    } else if (isSearchInnerNode(parentNode)) {
        // Parent is a search inner node - lock structural lock
        auto* searchParent = taggedCast<SearchInnerNode>(parentNode);
        parentLock = std::unique_lock<std::mutex>(searchParent->structural_lock_);
    }

    // Collect all data from the model node subtree
    std::vector<std::pair<KeyType, ValueType>> allData;
    collectAllData(tagPointer(modelNode, NodeType::ModelInner), allData);

    // Build new leaves from collected data
    std::vector<std::pair<KeyType, void*>> newLeaves = buildLeaves(std::move(allData));

    // Extract boundary keys for configuration search
    std::vector<KeyType> newBoundaryKeys;
    newBoundaryKeys.resize(newLeaves.size());
    std::transform(newLeaves.begin(), newLeaves.end(), newBoundaryKeys.begin(),
                   [](const std::pair<KeyType, void*>& p) { return p.first; });

    // Find optimal configuration
    ConfigurationSearch cs(newBoundaryKeys, lambda_);
    auto config = cs.search();

    // Create new model node with optimal parameters
    auto* newModelNode = new ModelInnerNode(
            config.best_slope,
            newBoundaryKeys.front(),
            config.best_mr
    );

    // Insert all leaves into the new model
    newModelNode->bulkLoad(newLeaves, config.best_slot_counts, lambda_);
    void* taggedNewModelNode = tagPointer(newModelNode, NodeType::ModelInner);

    // Update parent based on parent type (non-root cases)
    if (isModelInnerNode(parentNode)) {
        // Parent is a model inner node - use updateChildWithExternalLock
        auto* modelParent = taggedCast<ModelInnerNode>(parentNode);
        modelParent->updateChildWithExternalLock(newBoundaryKeys.front(), taggedNewModelNode);
    } else if (isSearchInnerNode(parentNode)) {
        // Parent is a search inner node - replace the child
        auto* searchParent = taggedCast<SearchInnerNode>(parentNode);
        searchParent->addChild(newBoundaryKeys.front(), taggedNewModelNode);
    }

    // Clean up old model node using safe deletion
    safeDelete(modelNode);
}

void Hyper::collectAllData(void* node, std::vector<std::pair<KeyType, ValueType>>& data) {
    if (!node) return;

    if (isLeafNode(node)) {
        auto* leaf = taggedCast<LeafNode>(node);
        auto leafData = leaf->gatherAll();
        data.insert(data.end(), leafData.begin(), leafData.end());

    } else if (isModelInnerNode(node)) {
        auto* modelNode = taggedCast<ModelInnerNode>(node);

        // Try to read with minimal locking
        size_t numSlots = modelNode->getNumSlots();
        std::set<void*> processedChildren;  // Avoid duplicates

        for (size_t i = 0; i < numSlots; i++) {
            std::mutex& slotLock = modelNode->getSlotLock(i);

            if (slotLock.try_lock()) {
                // Got the lock - can read safely
                const auto& slot = modelNode->getSlots()[i];
                if (slot.isRealChild() && processedChildren.find(slot.KeyChildPtr.second) == processedChildren.end()) {
                    processedChildren.insert(slot.KeyChildPtr.second);
                    collectAllData(slot.KeyChildPtr.second, data);
                }
                slotLock.unlock();
            } else {
                // Lock is held - use seq-lock read
                uint64_t version_before, version_after;
                void* child = nullptr;
                bool isReal = false;

                do {
                    version_before = modelNode->getSlotVersion(i);
                    if (version_before & 1) {
                        std::this_thread::yield();
                        continue;
                    }

                    const auto& slot = modelNode->getSlots()[i];
                    isReal = slot.isRealChild();
                    if (isReal) {
                        child = slot.KeyChildPtr.second;
                    }

                    version_after = modelNode->getSlotVersion(i);
                } while (version_before != version_after);

                if (isReal && child && processedChildren.find(child) == processedChildren.end()) {
                    processedChildren.insert(child);
                    collectAllData(child, data);
                }
            }
        }

    } else if (isSearchInnerNode(node)) {
        auto* searchNode = taggedCast<SearchInnerNode>(node);

        // For SearchInnerNode, use RCU read which is always safe
        auto children = searchNode->getChildren();
        for (const auto& [key, child] : children) {
            if (child) {
                collectAllData(child, data);
            }
        }
    }
}

void Hyper::convertSearchNodeToModelNode(SearchInnerNode* sNode, void* parentNode) {
    std::unique_lock<std::mutex> parentLock;
    KeyType searchNodeKey = 0; // Will store the key for this search node in parent

    // Parent is a model inner node - find and lock the appropriate slot
    auto* modelParent = taggedCast<ModelInnerNode>(parentNode);

    // Find which slot contains this search node
    size_t slotIdx = 0;
    void* taggedSearchNode = tagPointer(sNode, NodeType::SearchInner);

    for (size_t i = 0; i < modelParent->getNumSlots(); i++) {
        void* child = modelParent->getChildAtIndex(i);
        if (child == taggedSearchNode) {
            slotIdx = i;
            break;
        }
    }

    // Lock the specific slot
    parentLock = std::unique_lock<std::mutex>(modelParent->getSlotLock(slotIdx));

    // Get the key for this search node (needed for updating parent)
    const auto& slot = modelParent->getSlots()[slotIdx];
    if (slot.isRealChild()) {
        searchNodeKey = slot.KeyChildPtr.first;
    }

    // Collect all data
    std::vector<std::pair<KeyType, ValueType>> subtreeData;

   collectAllData(sNode, subtreeData);

    // Build new leaves from leaf data
    std::vector<std::pair<KeyType, void*>> newLeaves;
    newLeaves = buildLeaves(std::move(subtreeData));

    // Extract boundary keys for configuration search
    std::vector<KeyType> newBoundaryKeys;
    newBoundaryKeys.resize(newLeaves.size());
    std::transform(newLeaves.begin(), newLeaves.end(), newBoundaryKeys.begin(),
                   [](const std::pair<KeyType, void*>& p) { return p.first; });

    // Find optimal configuration
    ConfigurationSearch cs(newBoundaryKeys, lambda_);
    auto config = cs.search();

    // Create new model node with optimal parameters
    auto* newModelNode = new ModelInnerNode(
            config.best_slope,
            newBoundaryKeys.front(),
            config.best_mr
    );

    // Insert all leaves into the new model
    newModelNode->bulkLoad(newLeaves, config.best_slot_counts, lambda_);
    void* taggedNewModelNode = tagPointer(newModelNode, NodeType::ModelInner);

    // Update parent using updateChildWithExternalLock since we hold the slot lock
    modelParent->updateChildWithExternalLock(newBoundaryKeys.front(), taggedNewModelNode);

    // Clean up old search node using safe deletion
    safeDelete(sNode);
}

bool Hyper::updateRootRCU(void* oldRoot, void* newRoot) {
    // Attempt atomic compare-and-swap
    return root_.compare_exchange_weak(oldRoot, newRoot, 
                                      std::memory_order_release, 
                                      std::memory_order_acquire);
}

bool Hyper::needsRootUpdate(void* currentRoot, const std::pair<KeyType, void*>& leafDesc) {
    // Check if the key range of the new leaf falls outside current root's range
    if (isModelInnerNode(currentRoot)) {
        auto* modelRoot = taggedCast<ModelInnerNode>(currentRoot);
        KeyType rootMinKey = modelRoot->getMinKey();
        
        // If new leaf's key is smaller than root's min key, we need root update
        return leafDesc.first < rootMinKey;
    } else if (isSearchInnerNode(currentRoot)) {
        auto* searchRoot = taggedCast<SearchInnerNode>(currentRoot);
        auto children = searchRoot->getChildren();
        
        if (!children.empty()) {
            KeyType rootMinKey = children.front().first;
            return leafDesc.first < rootMinKey;
        }
    }
    
    return false;
}

InsertResult Hyper::handleRootUpdate(void* oldRoot, const std::vector<std::pair<KeyType, void*>>& splitDescriptors) {
    // This is a root split scenario - need to create new root
    std::lock_guard<std::mutex> rootGuard(root_update_lock_);
    
    // Double-check root hasn't changed
    void* currentRoot = root_.load(std::memory_order_acquire);
    if (currentRoot != oldRoot) {
        // Root changed, retry the entire operation
        return InsertResult::RetryFromRoot;
    }
    
    // Create new root to accommodate the split
    std::vector<std::pair<KeyType, void*>> allChildren;
    
    // Add existing root as a child
    if (isLeafNode(currentRoot)) {
        auto* leaf = taggedCast<LeafNode>(currentRoot);
        allChildren.emplace_back(leaf->getMinKey(), currentRoot);
    } else {
        // For inner nodes, we need to extract their min key
        KeyType minKey = 0;
        if (isModelInnerNode(currentRoot)) {
            minKey = taggedCast<ModelInnerNode>(currentRoot)->getMinKey();
        } else if (isSearchInnerNode(currentRoot)) {
            auto children = taggedCast<SearchInnerNode>(currentRoot)->getChildren();
            if (!children.empty()) {
                minKey = children.front().first;
            }
        }
        allChildren.emplace_back(minKey, currentRoot);
    }
    
    // Add split descriptors
    allChildren.insert(allChildren.end(), splitDescriptors.begin(), splitDescriptors.end());
    
    // Sort by key
    std::sort(allChildren.begin(), allChildren.end());
    
    // Create new root
    void* newRoot;
    if (allChildren.size() <= 8) {
        // Use search inner node
        auto* searchRoot = new SearchInnerNode();
        searchRoot->bulk_load(allChildren);
        newRoot = tagPointer(searchRoot, NodeType::SearchInner);
    } else {
        // Use model inner node with configuration search
        std::vector<KeyType> keys;
        keys.reserve(allChildren.size());
        std::transform(allChildren.begin(), allChildren.end(), std::back_inserter(keys),
                      [](const auto& p) { return p.first; });
        
        ConfigurationSearch cs(keys, lambda_);
        auto config = cs.search();
        
        auto* modelRoot = new ModelInnerNode(config.best_slope, keys.front(), config.best_mr);
        modelRoot->bulkLoad(allChildren, config.best_slot_counts, lambda_);
        newRoot = tagPointer(modelRoot, NodeType::ModelInner);
    }
    
    // Atomically update root
    root_.store(newRoot, std::memory_order_release);
    
    return InsertResult::Success;
}
