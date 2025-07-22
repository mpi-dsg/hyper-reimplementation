#include <iostream>
#include "../include/leaf_node.h"
#include "../include/epoch_manager.h"

// --- Slot implementation ---
LeafNode::Slot::Slot() {
    data.overflowPtr.store(nullptr, std::memory_order_relaxed);
}

LeafNode::Slot::~Slot() {
    destroy();
}

bool LeafNode::Slot::isEmpty() const {
    return !isKV() && data.overflowPtr.load(std::memory_order_acquire) == nullptr;
}

bool LeafNode::Slot::isKV() const {
    KeyType key = data.kv.key.load(std::memory_order_acquire);
    return (key & MSB_MASK) != 0;
}

bool LeafNode::Slot::isPointer() const {
    return !isKV() && data.overflowPtr.load(std::memory_order_acquire) != nullptr;
}

void LeafNode::Slot::destroy() {
    if (isPointer()) {
        OverflowBuffer* ptr = data.overflowPtr.load(std::memory_order_acquire);
        if (ptr) {
            delete ptr;
            data.overflowPtr.store(nullptr, std::memory_order_release);
        }
    }
}

void LeafNode::Slot::setSingle(KeyType k, ValueType v) {
    data.kv.key.store(k | MSB_MASK, std::memory_order_release);  // Set MSB to indicate this is a key-value pair
    data.kv.value = v;
}

void LeafNode::Slot::setOverflow(KeyType k, ValueType v) {
    OverflowBuffer* current = data.overflowPtr.load(std::memory_order_acquire);
    if (!current) {
        OverflowBuffer* newBuffer = new OverflowBuffer(4);  // Initial capacity of 4
        newBuffer->bulk_load({{k, v}});
        data.overflowPtr.store(newBuffer, std::memory_order_release);
    } else {
        // RCU: Create new buffer with inserted element
        OverflowBuffer* newBuffer = current->insertRCU(k, v);
        data.overflowPtr.store(newBuffer, std::memory_order_release);
        // Schedule old buffer for safe deletion
        safeDelete(current);
    }
}

// --- LeafNode implementation ---
LeafNode::LeafNode(double slope, KeyType minKey, KeyType maxKey)
        : slope_(slope), minKey_(minKey), maxPossibleKey_(std::numeric_limits<KeyType>::max()),
          MR_(ceil(slope * static_cast<double>(maxKey - minKey))), slots_(MR_+1),
          init_histogram_(MR_+1, 0), op_counter_(0) {}

LeafNode::LeafNode(const LeafNode& other)
        : slope_(other.slope_), minKey_(other.minKey_), maxPossibleKey_(other.maxPossibleKey_),
          MR_(other.MR_), slots_(other.MR_ + 1), init_histogram_(other.init_histogram_),
          op_counter_(other.op_counter_.load()) {

    // Deep copy all slots (locks are initialized fresh)
    for (size_t i = 0; i < other.slots_.size(); ++i) {
        const auto& srcSlot = other.slots_[i];
        if (srcSlot.isKV()) {
            KeyType key = srcSlot.data.kv.key.load(std::memory_order_acquire);
            ValueType value = srcSlot.data.kv.value;
            slots_[i].data.kv.key.store(key, std::memory_order_release);
            slots_[i].data.kv.value = value;
        } else if (srcSlot.isPointer()) {
            OverflowBuffer* srcBuffer = srcSlot.data.overflowPtr.load(std::memory_order_acquire);
            if (srcBuffer) {
                OverflowBuffer* newBuffer = new OverflowBuffer(*srcBuffer);
                slots_[i].data.overflowPtr.store(newBuffer, std::memory_order_release);
            }
        }
    }
}

LeafNode::~LeafNode() {
    for (auto& slot : slots_) {
        slot.destroy();
    }
}

InsertReturn LeafNode::insert(
        KeyType key,
        ValueType value,
        double delta) {

    // Check if key exceeds maximum possible key for this leaf
    if (key >= maxPossibleKey_) {
        // Return retry signal instead of throwing exception
        return InsertReturn(InsertResult::RetryFromRoot);
    }

    // Predict the slot for the key
    size_t idx = predictSlot(key);
    bool needsSplit = false;

    // Lock only the specific slot we're inserting into
    std::unique_lock<std::mutex> slotLock(slots_[idx].lock);
    auto& slot = slots_[idx];

    // Handle different scenarios based on slot state
    if (slot.isEmpty()) {
        // Empty slot - simply insert the key-value pair
        slot.setSingle(key, value);
    } else if (slot.isKV()) {
        // Slot contains a key-value pair - convert to overflow buffer
        KeyType existingKey = decodeKey(slot.data.kv.key.load(std::memory_order_acquire));
        ValueType existingValue = slot.data.kv.value;

        auto p0 = std::make_pair(existingKey, existingValue);
        auto p1 = std::make_pair(key, value);

        if (p1.first < p0.first) {
            std::swap(p0, p1);
        }

        // Create new overflow buffer with both elements
        OverflowBuffer* newBuffer = new OverflowBuffer(4);
        newBuffer->bulk_load({p0, p1});

        // Atomically update the slot to point to overflow buffer
        slot.data.overflowPtr.store(newBuffer, std::memory_order_release);
    } else {
        // Slot contains an overflow buffer - add the new pair using RCU
        OverflowBuffer* current = slot.data.overflowPtr.load(std::memory_order_acquire);
        OverflowBuffer* newBuffer = current->insertRCU(key, value);
        slot.data.overflowPtr.store(newBuffer, std::memory_order_release);
        
        // Schedule old buffer for safe deletion
        safeDelete(current);

        // Trigger segmentation if overflow buffer becomes too large
        if (newBuffer->size() > (2 * delta + 1)) {
            needsSplit = true;
        }
    }

    // Release the slot lock before checking for split
    slotLock.unlock();

    // Increment operation counter
    op_counter_.fetch_add(1, std::memory_order_relaxed);

    if (!needsSplit) {
        return InsertReturn(InsertResult::Success);  // No restructuring needed
    }

    // Check if we need to perform split with parent lock
    auto splitResult = performSplitWithParentLock(delta);
    if (splitResult.has_value()) {
        return InsertReturn(InsertResult::SuccessWithSplit, std::move(*splitResult));
    }

    return InsertReturn(InsertResult::Success);  // Split was attempted but not completed
}

std::optional<ValueType> LeafNode::find(KeyType key) const {
    // key cannot be in this node
    if (key > maxPossibleKey_) {
        return std::numeric_limits<ValueType>::max();
    }

    size_t idx = predictSlot(key);
    const auto& slot = slots_[idx];

    // Lock-free read - use atomic loads for consistency
    if (slot.isPointer()) {
        // Key is in an overflow buffer - search there
        OverflowBuffer* buffer = slot.data.overflowPtr.load(std::memory_order_acquire);
        if (buffer) {
            return buffer->find(key);
        }
        return std::nullopt;
    } else if (slot.isKV()) {
        // Slot contains a direct key-value pair
        KeyType stored_key = decodeKey(slot.data.kv.key.load(std::memory_order_acquire));
        if (stored_key == key) {
            return slot.data.kv.value;
        }
        return std::nullopt;
    } else {
        // Slot is empty
        return std::nullopt;
    }
}

bool LeafNode::erase(KeyType key) {
    size_t idx = predictSlot(key);
    auto& slot = slots_[idx];

    // Lock the specific slot for erase operation
    std::lock_guard<std::mutex> slotLock(slot.lock);

    if (!slot.isPointer() && slot.isKV()) {
        KeyType original_key = decodeKey(slot.data.kv.key.load(std::memory_order_acquire));
        if (original_key == key) {
            slot.destroy();
            return true;
        }
        return false;
    } else if (slot.isPointer()) {
        OverflowBuffer* current = slot.data.overflowPtr.load(std::memory_order_acquire);
        if (current) {
            OverflowBuffer* newBuffer = current->eraseRCU(key);
            if (newBuffer) {
                slot.data.overflowPtr.store(newBuffer, std::memory_order_release);
                // Schedule old buffer for safe deletion
                safeDelete(current);
                return true;
            }
        }
    }
    return false;
}

void LeafNode::bulkLoad(std::vector<std::pair<KeyType, ValueType>>&& data) {
    // Initialize histograms and counters
    init_histogram_.assign(slots_.size(), 0);
    op_counter_.store(0, std::memory_order_relaxed);

    size_t n = data.size();
    size_t i = 0;

    while (i < n) {
        // Determine the slot index for the current key
        size_t idx = predictSlot(data[i].first);
        size_t start = i;

        // Process all consecutive elements that map to the same slot
        while (i < n && predictSlot(data[i].first) == idx) {
            i++;
        }

        // Create a group of elements for this slot
        std::vector<std::pair<KeyType, ValueType>> group(
                std::make_move_iterator(data.begin() + start),
                std::make_move_iterator(data.begin() + i)
        );

        // Store the group in the slot
        if (group.size() == 1) {
            // Direct storage for a single key-value pair
            slots_[idx].setSingle(group[0].first, group[0].second);
            init_histogram_[idx] = 1;
        } else {
            // Use overflow buffer for multiple key-value pairs
            OverflowBuffer* buffer = new OverflowBuffer(group.size());
            init_histogram_[idx] = group.size();
            buffer->bulk_load(std::move(group));
            slots_[idx].data.overflowPtr.store(buffer, std::memory_order_release);
        }
    }
}

std::vector<std::pair<KeyType, ValueType>> LeafNode::gatherAll() const {
    // Count the total number of key-value pairs
    size_t total = 0;
    for (const auto& s : slots_) {
        if (s.isKV())
            total += 1;
        else if (s.isPointer()) {
            OverflowBuffer* buffer = s.data.overflowPtr.load(std::memory_order_acquire);
            if (buffer) total += buffer->size();
        }
    }

    // Gather all key-value pairs
    std::vector<std::pair<KeyType, ValueType>> all;
    all.reserve(total);

    for (const auto& s : slots_) {
        if (s.isKV()) {
            all.emplace_back(decodeKey(s.data.kv.key.load(std::memory_order_acquire)), s.data.kv.value);
        } else if (s.isPointer()) {
            OverflowBuffer* buffer = s.data.overflowPtr.load(std::memory_order_acquire);
            if (buffer) {
                const auto& vec = buffer->data();
                all.insert(all.end(), vec.begin(), vec.end());
            }
        }
    }
    return all;
}

KeyType LeafNode::decodeKey(KeyType stored_key) const {
    KeyType keyWithoutMSB = stored_key & ~MSB_MASK;
    if (keyWithoutMSB < minKey_) {
        return stored_key;
    }
    return keyWithoutMSB;
}

std::optional<std::vector<std::pair<KeyType, void*>>> LeafNode::performSplitWithParentLock(double delta) {
    // Wait until all leaf node slots are free
    // This is a busy-wait loop, but splits should be rare
    while (true) {
        // Try to acquire all slot locks
        auto allSlotLocks = tryLockAllSlots();
        if (!allSlotLocks.empty()) {
            // Successfully acquired all locks, proceed with split
            auto data = gatherAll();
            auto result = performSplit(data, delta);
            // Slot locks automatically released when allSlotLocks goes out of scope
            return result;
        }

        // Some slots are still locked, yield and try again
        std::this_thread::yield();
    }
}

bool LeafNode::hasAnySlotLocked() const {
    for (const auto& slot : slots_) {
        if (slot.lock.try_lock()) {
            // Could acquire lock, so it wasn't locked, unlock immediately
            slot.lock.unlock();
        } else {
            // Couldn't acquire lock, so it's locked by another thread
            return true;
        }
    }
    return false;
}

std::vector<std::unique_lock<std::mutex>> LeafNode::tryLockAllSlots() {
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(slots_.size());

    // Try to acquire all locks
    for (auto& slot : slots_) {
        std::unique_lock<std::mutex> lock(slot.lock, std::try_to_lock);
        if (!lock.owns_lock()) {
            // Failed to acquire a lock, release all previously acquired locks
            locks.clear();
            return locks; // Return empty vector to indicate failure
        }
        locks.push_back(std::move(lock));
    }

    return locks; // Return all locks if successful
}

std::vector<std::pair<KeyType, void*>> LeafNode::performSplit(
        const std::vector<std::pair<KeyType, ValueType>>& data,
        double delta) {

    // Generate piece-wise linear segments for the data
    std::vector<Hyper::PLASegment> segments;
    KeyType maxKey = data.back().first;

    hyperpgm::internal::make_segmentation_par(
            data.size(),
            static_cast<size_t>(delta),
            [&](size_t i) { return data[i].first; },
            [&](const auto& cs) {
                Hyper::PLASegment seg{};
                seg.min_key = cs.get_first_x();
                seg.max_key = cs.get_last_x();
                if (cs.get_last_x() >= maxKey)
                    seg.max_key = data.back().first;
                auto [segSlope, intercept] = cs.get_floating_point_segment(seg.min_key);
                seg.slope = segSlope;
                auto start_it = std::lower_bound(data.begin(), data.end(), seg.min_key,
                                                 [](const auto& pair, KeyType key) { return pair.first < key; });
                seg.start_idx = std::distance(data.begin(), start_it);
                auto end_it = std::lower_bound(data.begin(), data.end(), seg.max_key,
                                               [](const auto& pair, KeyType key) { return pair.first < key; });
                seg.end_idx = std::distance(data.begin(), end_it);
                segments.push_back(seg);
            }
    );

    // Check if the last segment is valid, remove it if not
    if (!segments.empty() && segments.back().start_idx >= data.size()) {
        segments.pop_back();
    }

    std::vector<std::pair<KeyType, void*>> newLeaves;
    newLeaves.reserve(segments.size());

    // Split the node into multiple nodes
    for (size_t i = 0; i < segments.size(); ++i) {
        auto seg = segments[i];
        std::vector<std::pair<KeyType, ValueType>> segData(
                std::make_move_iterator(const_cast<std::pair<KeyType, ValueType>*>(&data[seg.start_idx])),
                std::make_move_iterator(const_cast<std::pair<KeyType, ValueType>*>(&data[seg.end_idx + 1]))
        );

        LeafNode* segLeaf = new LeafNode(seg.slope, seg.min_key, seg.max_key);
        if (i < segments.size() - 1) {
            segLeaf->maxPossibleKey_ = segments[i + 1].min_key - 1;
        }
        segLeaf->bulkLoad(std::move(segData));

        void* taggedLeaf = tagPointer(segLeaf, NodeType::Leaf);
        newLeaves.emplace_back(seg.min_key, taggedLeaf);
    }

    // Return new leaves in reverse order
    std::reverse(newLeaves.begin(), newLeaves.end());

    return newLeaves;
}

bool LeafNode::checkPolicyTwo() {
    // Create a histogram of the current data distribution
    std::vector<int> current_hist(slots_.size(), 0);
    for (size_t i = 0; i < slots_.size(); i++) {
        const auto& s = slots_[i];
        int count = 0;
        if (s.isKV())
            count = 1;
        else if (s.isPointer()) {
            OverflowBuffer *buffer = s.data.overflowPtr.load(std::memory_order_acquire);
            if (buffer) count = buffer->size();
        }
        current_hist[i] = count;
    }

    // Calculate Kolmogorov-Smirnov statistic
    int sup_diff = 0;
    int cumulative_init = 0;
    int cumulative_current = 0;
    for (size_t i = 0; i < current_hist.size(); i++) {
        cumulative_init += init_histogram_[i];
        cumulative_current += current_hist[i];
        int diff = std::abs(cumulative_current - cumulative_init);
        if (diff > sup_diff)
            sup_diff = diff;
    }

    // Calculate sample sizes and threshold for significance
    int n = 0, m = 0;
    for (int x : init_histogram_) n += x;
    for (int x : current_hist) m += x;
    double threshold = 1.731 * std::sqrt((n + m) / static_cast<double>(n * m));

    // Reset operation counter
    op_counter_.store(0, std::memory_order_relaxed);

    // Return true if distribution shift is significant
    return sup_diff > threshold;
}
