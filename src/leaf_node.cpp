#include <iostream>
#include <cmath>
#include <algorithm>
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
    // Clear KV marker so the slot becomes empty (needed for erase of accurate slots).
    data.kv.key.store(0, std::memory_order_release);
    data.kv.value = 0;
}

void LeafNode::Slot::setSingle(KeyType k, ValueType v) {
    data.kv.key.store(k | MSB_MASK, std::memory_order_release);  // Set MSB to indicate this is a key-value pair
    data.kv.value = v;
}

void LeafNode::Slot::setOverflow(KeyType k, ValueType v) {
    OverflowBuffer* current = data.overflowPtr.load(std::memory_order_acquire);
    if (!current) {
        OverflowBuffer* newBuffer = new OverflowBuffer(4);
        newBuffer->bulk_load({{k, v}});
        data.overflowPtr.store(newBuffer, std::memory_order_release);
        return;
    }
    if (!isHyperLockingEnabled()) {
        current->insert(k, v);
        return;
    }
    OverflowBuffer* newBuffer = current->insertRCU(k, v);
    data.overflowPtr.store(newBuffer, std::memory_order_release);
    safeDelete(current);
}

// --- LeafNode implementation ---
LeafNode::LeafNode(double slope, KeyType minKey, KeyType maxKey)
        : slope_(slope), minKey_(minKey), maxPossibleKey_(std::numeric_limits<KeyType>::max()),
          MR_(ceil(slope * static_cast<double>(maxKey - minKey))), slots_(MR_+1),
          op_counter_(0), init_histogram_(MR_+1, 0) {}

LeafNode::LeafNode(const LeafNode& other)
        : slope_(other.slope_), minKey_(other.minKey_), maxPossibleKey_(other.maxPossibleKey_),
          MR_(other.MR_), slots_(other.MR_ + 1),
          op_counter_(other.op_counter_.load()), init_histogram_(other.init_histogram_) {

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

size_t LeafNode::slotConflictCount(size_t idx) const {
    if (idx >= slots_.size()) return 0;
    const auto& s = slots_[idx];
    if (s.isKV()) return 1;
    if (s.isPointer()) {
        OverflowBuffer* buffer = s.data.overflowPtr.load(std::memory_order_acquire);
        return buffer ? buffer->size() : 0;
    }
    return 0;
}

size_t LeafNode::maxConflictCount() const {
    size_t peak = 0;
    for (size_t i = 0; i < slots_.size(); ++i) {
        peak = std::max(peak, slotConflictCount(i));
    }
    return peak;
}

void LeafNode::overflowInsert(Slot& slot, KeyType key, ValueType value) {
    OverflowBuffer* current = slot.data.overflowPtr.load(std::memory_order_acquire);
    if (!current) {
        OverflowBuffer* neu = new OverflowBuffer(4);
        neu->bulk_load({{key, value}});
        slot.data.overflowPtr.store(neu, std::memory_order_release);
        return;
    }
    if (!isHyperLockingEnabled()) {
        current->insert(key, value);
        return;
    }
    OverflowBuffer* neu = current->insertRCU(key, value);
    slot.data.overflowPtr.store(neu, std::memory_order_release);
    safeDelete(current);
}

bool LeafNode::overflowErase(Slot& slot, KeyType key) {
    OverflowBuffer* current = slot.data.overflowPtr.load(std::memory_order_acquire);
    if (!current) return false;

    if (!isHyperLockingEnabled()) {
        if (!current->erase(key)) return false;
        if (current->size() == 0) {
            delete current;
            slot.data.overflowPtr.store(nullptr, std::memory_order_release);
            slot.data.kv.key.store(0, std::memory_order_release);
            slot.data.kv.value = 0;
        } else if (current->size() == 1) {
            auto only = current->data().front();
            delete current;
            slot.data.overflowPtr.store(nullptr, std::memory_order_release);
            slot.setSingle(only.first, only.second);
        }
        return true;
    }

    OverflowBuffer* neu = current->eraseRCU(key);
    if (!neu) return false;
    if (neu->size() == 0) {
        delete neu;
        slot.data.overflowPtr.store(nullptr, std::memory_order_release);
        slot.data.kv.key.store(0, std::memory_order_release);
        slot.data.kv.value = 0;
    } else if (neu->size() == 1) {
        auto only = neu->data().front();
        delete neu;
        slot.data.overflowPtr.store(nullptr, std::memory_order_release);
        slot.setSingle(only.first, only.second);
    } else {
        slot.data.overflowPtr.store(neu, std::memory_order_release);
    }
    safeDelete(current);
    return true;
}

InsertReturn LeafNode::insert(
        KeyType key,
        ValueType value,
        double delta) {

    if (key >= maxPossibleKey_) {
        return InsertReturn(InsertResult::RetryFromRoot);
    }

    size_t idx = predictSlot(key);
    bool needsSplit = false;

    std::unique_lock<std::mutex> slotLock(slots_[idx].lock, std::defer_lock);
    if (isHyperLockingEnabled()) {
        slotLock.lock();
    }
    auto& slot = slots_[idx];

    if (slot.isEmpty()) {
        slot.setSingle(key, value);
    } else if (slot.isKV()) {
        KeyType existingKey = decodeKey(slot.data.kv.key.load(std::memory_order_acquire));
        ValueType existingValue = slot.data.kv.value;
        if (existingKey == key) {
            slot.data.kv.value = value;
        } else {
            auto p0 = std::make_pair(existingKey, existingValue);
            auto p1 = std::make_pair(key, value);
            if (p1.first < p0.first) std::swap(p0, p1);
            // Clear KV tag before storing overflow pointer (union aliasing).
            slot.data.kv.key.store(0, std::memory_order_relaxed);
            OverflowBuffer* newBuffer = new OverflowBuffer(4);
            newBuffer->bulk_load({p0, p1});
            slot.data.overflowPtr.store(newBuffer, std::memory_order_release);
        }
    } else {
        overflowInsert(slot, key, value);
    }

    size_t conflicts = slotConflictCount(idx);
    // Policy 1: C^max_leaf, and Corollary 3.1 overflow bound.
    if (conflicts >= kMaxLeafConflicts || conflicts > static_cast<size_t>(2 * delta + 1)) {
        needsSplit = true;
    }

    if (slotLock.owns_lock()) {
        slotLock.unlock();
    }

    // Policy 2: check KS divergence when the 16-bit op counter wraps.
    uint16_t prev = op_counter_.fetch_add(1, std::memory_order_relaxed);
    if (static_cast<uint16_t>(prev + 1) == 0 && checkPolicyTwo()) {
        needsSplit = true;
    }

    if (!needsSplit) {
        return InsertReturn(InsertResult::Success);
    }

    auto splitResult = performSplitWithParentLock(delta);
    if (splitResult.has_value()) {
        return InsertReturn(InsertResult::SuccessWithSplit, std::move(*splitResult));
    }

    return InsertReturn(InsertResult::Success);
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
    MaybeLock slotLock(slot.lock);

    // Paper §4.4: deleting the leftmost key must not change minKey_ metadata.

    if (!slot.isPointer() && slot.isKV()) {
        KeyType original_key = decodeKey(slot.data.kv.key.load(std::memory_order_acquire));
        if (original_key == key) {
            slot.destroy();
            return true;
        }
        return false;
    }
    if (slot.isPointer()) {
        return overflowErase(slot, key);
    }
    return false;
}

bool LeafNode::update(KeyType key, ValueType value) {
    if (key > maxPossibleKey_) return false;
    size_t idx = predictSlot(key);
    auto& slot = slots_[idx];
    MaybeLock slotLock(slot.lock);

    if (slot.isKV()) {
        KeyType original_key = decodeKey(slot.data.kv.key.load(std::memory_order_acquire));
        if (original_key != key) return false;
        slot.data.kv.value = value;
        return true;
    }
    if (slot.isPointer()) {
        OverflowBuffer* buf = slot.data.overflowPtr.load(std::memory_order_acquire);
        if (!buf || !buf->find(key).has_value()) return false;
        if (!isHyperLockingEnabled()) {
            buf->insert(key, value);
            return true;
        }
        OverflowBuffer* neu = buf->insertRCU(key, value);
        slot.data.overflowPtr.store(neu, std::memory_order_release);
        safeDelete(buf);
        return true;
    }
    return false;
}

size_t LeafNode::size() const {
    size_t n = 0;
    for (const auto& slot : slots_) {
        if (slot.isKV()) {
            ++n;
        } else if (slot.isPointer()) {
            OverflowBuffer* buffer = slot.data.overflowPtr.load(std::memory_order_acquire);
            if (buffer) n += buffer->size();
        }
    }
    return n;
}

double LeafNode::density() const {
    size_t cap = capacity();
    if (cap == 0) return 0.0;
    return static_cast<double>(size()) / static_cast<double>(cap);
}

bool LeafNode::maybeRebuildLowDensity(double min_density) {
    if (density() >= min_density) return false;
    auto data = gatherAll();
    if (data.empty()) return false;

    // Rebuild in place: clear slots then bulk-load. Keeps slope_/minKey_/MR_.
    for (auto& slot : slots_) {
        slot.destroy();
    }
    bulkLoad(std::move(data));
    return true;
}

size_t LeafNode::memoryBytes() const {
    size_t bytes = sizeof(LeafNode);
    bytes += slots_.capacity() * sizeof(Slot);
    bytes += init_histogram_.capacity() * sizeof(int);
    for (const auto& slot : slots_) {
        if (slot.isPointer()) {
            OverflowBuffer* buffer = slot.data.overflowPtr.load(std::memory_order_acquire);
            if (buffer) {
                bytes += sizeof(OverflowBuffer);
                bytes += buffer->size() * sizeof(std::pair<KeyType, ValueType>);
            }
        }
    }
    return bytes;
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
    // Paper §6.1 ST mode: skip fine-grained locking.
    if (!isHyperLockingEnabled()) {
        auto data = gatherAll();
        return performSplit(data, delta);
    }

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

    int n = 0, m = 0;
    for (int x : init_histogram_) n += x;
    for (int x : current_hist) m += x;
    if (n <= 0 || m <= 0) {
        op_counter_.store(0, std::memory_order_relaxed);
        return false;
    }

    // Two-sample KS on slot histograms; beta=0.005 (paper §3.3.1).
    double d_stat = 0.0;
    int cumulative_init = 0;
    int cumulative_current = 0;
    for (size_t i = 0; i < current_hist.size(); i++) {
        cumulative_init += init_histogram_[i];
        cumulative_current += current_hist[i];
        double diff = std::abs(static_cast<double>(cumulative_current) / m -
                               static_cast<double>(cumulative_init) / n);
        if (diff > d_stat) d_stat = diff;
    }

    const double c_beta = std::sqrt(-0.5 * std::log(0.005 / 2.0));
    double threshold = c_beta * std::sqrt((n + m) / static_cast<double>(n * m));

    op_counter_.store(0, std::memory_order_relaxed);
    return d_stat > threshold;
}
