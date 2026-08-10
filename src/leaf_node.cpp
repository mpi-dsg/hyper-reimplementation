#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "../include/leaf_node.h"
#include "../include/epoch_manager.h"

// --- Slot implementation (16B plain cells; locks live on the leaf) ---
LeafNode::Slot::Slot() : key(0), overflowPtr(nullptr) {}

LeafNode::Slot::~Slot() {
    destroy();
}

bool LeafNode::Slot::isEmpty() const {
    return !isKV() && overflowPtr == nullptr;
}

bool LeafNode::Slot::isKV() const {
    return (key & MSB_MASK) != 0;
}

bool LeafNode::Slot::isPointer() const {
    return !isKV() && overflowPtr != nullptr;
}

void LeafNode::Slot::destroy() {
    if (isPointer()) {
        OverflowBuffer* ptr = overflowPtr;
        if (ptr) {
            delete ptr;
            overflowPtr = nullptr;
        }
    }
    key = 0;
    value = 0;
}

void LeafNode::Slot::setSingle(KeyType k, ValueType v) {
    key = k | MSB_MASK;
    value = v;
}

void LeafNode::Slot::setOverflow(KeyType k, ValueType v) {
    OverflowBuffer* current = overflowPtr;
    if (!current) {
        OverflowBuffer* newBuffer = new OverflowBuffer(4);
        newBuffer->bulk_load({{k, v}});
        key = 0;
        overflowPtr = newBuffer;
        return;
    }
    if (!isHyperLockingEnabled()) {
        current->insert(k, v);
        return;
    }
    OverflowBuffer* newBuffer = current->insertRCU(k, v);
    key = 0;
    overflowPtr = newBuffer;
    safeDelete(current);
}

// --- LeafNode implementation ---
LeafNode::LeafNode(double slope, KeyType minKey, KeyType maxKey)
        : slope_(slope), minKey_(minKey), maxPossibleKey_(std::numeric_limits<KeyType>::max()),
          MR_(ceil(slope * static_cast<double>(maxKey - minKey))), slots_(MR_+1),
          op_counter_ptr_(0), op_counter_st_(0) {
    if (isHyperLockingEnabled()) {
        slot_locks_ = std::make_unique<HyperSlotMutex[]>(MR_ + 1);
    }
}

LeafNode::LeafNode(const LeafNode& other)
        : slope_(other.slope_), minKey_(other.minKey_), maxPossibleKey_(other.maxPossibleKey_),
          MR_(other.MR_), slots_(other.MR_ + 1),
          op_counter_ptr_(other.op_counter_ptr_.load()), op_counter_st_(other.op_counter_st_),
          init_histogram_len_(other.init_histogram_len_) {
    if (isHyperLockingEnabled()) {
        slot_locks_ = std::make_unique<HyperSlotMutex[]>(MR_ + 1);
    }
    if (other.init_histogram_ && other.init_histogram_len_) {
        init_histogram_ = std::make_unique<uint16_t[]>(other.init_histogram_len_);
        std::copy(other.init_histogram_.get(),
                  other.init_histogram_.get() + other.init_histogram_len_,
                  init_histogram_.get());
    }
    // Deep copy all slots
    for (size_t i = 0; i < other.slots_.size(); ++i) {
        const auto& srcSlot = other.slots_[i];
        if (srcSlot.isKV()) {
            KeyType key = srcSlot.key;
            ValueType value = srcSlot.value;
            slots_[i].key = key;
            slots_[i].value = value;
        } else if (srcSlot.isPointer()) {
            OverflowBuffer* srcBuffer = srcSlot.overflowPtr;
            if (srcBuffer) {
                OverflowBuffer* newBuffer = new OverflowBuffer(*srcBuffer);
                slots_[i].key = 0;
                slots_[i].overflowPtr = newBuffer;
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
        OverflowBuffer* buffer = s.overflowPtr;
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
    OverflowBuffer* current = slot.overflowPtr;
    if (!current) {
        OverflowBuffer* neu = new OverflowBuffer(4);
        neu->bulk_load({{key, value}});
        slot.key = 0;
        slot.overflowPtr = neu;
        return;
    }
    if (!isHyperLockingEnabled()) {
        current->insert(key, value);
        return;
    }
    OverflowBuffer* neu = current->insertRCU(key, value);
    slot.key = 0;
        slot.overflowPtr = neu;
    safeDelete(current);
}

bool LeafNode::overflowErase(Slot& slot, KeyType key) {
    OverflowBuffer* current = slot.overflowPtr;
    if (!current) return false;

    if (!isHyperLockingEnabled()) {
        if (!current->erase(key)) return false;
        if (current->size() == 0) {
            delete current;
            slot.overflowPtr = nullptr;
            slot.key = 0;
            slot.key = 0;
            slot.value = 0;
        } else if (current->size() == 1) {
            auto only = current->data().front();
            delete current;
            slot.overflowPtr = nullptr;
            slot.key = 0;
            slot.setSingle(only.first, only.second);
        }
        return true;
    }

    OverflowBuffer* neu = current->eraseRCU(key);
    if (!neu) return false;
    if (neu->size() == 0) {
        delete neu;
        slot.overflowPtr = nullptr;
            slot.key = 0;
        slot.key = 0;
        slot.value = 0;
    } else if (neu->size() == 1) {
        auto only = neu->data().front();
        delete neu;
        slot.overflowPtr = nullptr;
            slot.key = 0;
        slot.setSingle(only.first, only.second);
    } else {
        slot.key = 0;
        slot.overflowPtr = neu;
    }
    safeDelete(current);
    return true;
}

InsertReturn LeafNode::insert(
        KeyType key,
        ValueType value,
        double delta) {

    // maxPossibleKey_ is inclusive (next sibling min - 1). Match find/erase.
    if (key > maxPossibleKey_) {
        return InsertReturn(InsertResult::RetryFromRoot);
    }

    size_t idx = predictSlot(key);
    bool needsSplit = false;

    std::unique_lock<HyperSlotMutex> guard;
    if (hasSlotLocks()) {
        guard = std::unique_lock<HyperSlotMutex>(getSlotMutex(idx));
    }
    auto& slot = slots_[idx];

    if (slot.isEmpty()) {
        slot.setSingle(key, value);
    } else if (slot.isKV()) {
        KeyType existingKey = decodeKey(slot.key);
        ValueType existingValue = slot.value;
        if (existingKey == key) {
            slot.value = value;
        } else {
            auto p0 = std::make_pair(existingKey, existingValue);
            auto p1 = std::make_pair(key, value);
            if (p1.first < p0.first) std::swap(p0, p1);
            // Clear KV tag before storing overflow pointer (union aliasing).
            slot.key = 0;
            OverflowBuffer* newBuffer = new OverflowBuffer(4);
            newBuffer->bulk_load({p0, p1});
            slot.key = 0;
            slot.overflowPtr = newBuffer;
        }
    } else {
        overflowInsert(slot, key, value);
    }

    size_t conflicts = slotConflictCount(idx);
    // Corollary 3.1 hard bound requires a split.
    if (conflicts > static_cast<size_t>(2 * delta + 1)) {
        needsSplit = true;
    }
    // Policy 1: try cheap in-place retrain when C^max_leaf is exceeded.
    bool needsRetrain = !needsSplit && conflicts >= kMaxLeafConflicts;

    if (guard.owns_lock()) {
        guard.unlock();
    }

    // Policy 2: 16-bit counter in pointer word; wrap triggers KS check.
    if (bumpOpCounterWrapped() && checkPolicyTwo()) {
        needsRetrain = true;
    }

    if (needsRetrain && !needsSplit) {
        if (tryRetrainInPlace(delta)) {
            return InsertReturn(InsertResult::Success);
        }
        needsSplit = true;
    }

    if (!needsSplit) {
        return InsertReturn(InsertResult::Success);
    }

    // §4.2.2: hold SMO lock so readers still see this leaf until parents updated.
    if (isHyperLockingEnabled()) {
        smo_lock_.lock();
        smo_held_ = true;
    }

    auto splitResult = performSplitWithParentLock(delta);
    if (splitResult.has_value()) {
        return InsertReturn(InsertResult::SuccessWithSplit, std::move(*splitResult),
                            smo_held_ ? this : nullptr);
    }
    if (smo_held_) {
        endSmo();
    }
    return InsertReturn(InsertResult::Success);
}

bool LeafNode::bumpOpCounterWrapped() {
    if (!isHyperLockingEnabled()) {
        const uint16_t before = op_counter_st_++;
        return before == 0xFFFF;
    }
    const uintptr_t prev = op_counter_ptr_.fetch_add(
        uintptr_t(1) << kOpCounterShift, std::memory_order_relaxed);
    const uint16_t before = static_cast<uint16_t>(prev >> kOpCounterShift);
    return before == 0xFFFF;
}

void LeafNode::ensureInitHistogram(size_t n) {
    if (!init_histogram_ || init_histogram_len_ != n) {
        init_histogram_ = std::make_unique<uint16_t[]>(n);
        init_histogram_len_ = n;
    }
    std::fill(init_histogram_.get(), init_histogram_.get() + n, 0);
}

void LeafNode::endSmo() {
    if (smo_held_) {
        smo_held_ = false;
        smo_lock_.unlock();
    }
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
        OverflowBuffer* buffer = slot.overflowPtr;
        if (buffer) {
            return buffer->find(key);
        }
        return std::nullopt;
    } else if (slot.isKV()) {
        // Slot contains a direct key-value pair
        KeyType stored_key = decodeKey(slot.key);
        if (stored_key == key) {
            return slot.value;
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
    std::unique_lock<HyperSlotMutex> guard;
    if (hasSlotLocks()) {
        guard = std::unique_lock<HyperSlotMutex>(getSlotMutex(idx));
    }

    // Paper §4.4: deleting the leftmost key must not change minKey_ metadata.

    if (!slot.isPointer() && slot.isKV()) {
        KeyType original_key = decodeKey(slot.key);
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
    std::unique_lock<HyperSlotMutex> guard;
    if (hasSlotLocks()) {
        guard = std::unique_lock<HyperSlotMutex>(getSlotMutex(idx));
    }

    if (slot.isKV()) {
        KeyType original_key = decodeKey(slot.key);
        if (original_key != key) return false;
        slot.value = value;
        return true;
    }
    if (slot.isPointer()) {
        OverflowBuffer* buf = slot.overflowPtr;
        if (!buf || !buf->find(key).has_value()) return false;
        if (!isHyperLockingEnabled()) {
            buf->insert(key, value);
            return true;
        }
        OverflowBuffer* neu = buf->insertRCU(key, value);
        slot.key = 0;
        slot.overflowPtr = neu;
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
            OverflowBuffer* buffer = slot.overflowPtr;
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
    bytes += init_histogram_len_ * sizeof(uint16_t);
    if (slot_locks_) bytes += (MR_ + 1) * sizeof(HyperSlotMutex);
    for (const auto& slot : slots_) {
        if (slot.isPointer()) {
            OverflowBuffer* buffer = slot.overflowPtr;
            if (buffer) {
                // sizeof(OverflowBuffer) already includes inline SSO storage;
                // only add heap payload beyond that.
                bytes += sizeof(OverflowBuffer);
                if (buffer->size() > OverflowBuffer::kInlineCap) {
                    bytes += buffer->size() * sizeof(std::pair<KeyType, ValueType>);
                }
            }
        }
    }
    return bytes;
}

void LeafNode::bulkLoad(std::vector<std::pair<KeyType, ValueType>>&& data) {
    ensureInitHistogram(slots_.size());
    op_counter_ptr_.store(0, std::memory_order_relaxed);
    op_counter_st_ = 0;

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
            init_histogram_[idx] = static_cast<uint16_t>(std::min(group.size(), size_t(65535)));
            buffer->bulk_load(std::move(group));
            slots_[idx].key = 0;
            slots_[idx].overflowPtr = buffer;
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
            OverflowBuffer* buffer = s.overflowPtr;
            if (buffer) total += buffer->size();
        }
    }

    // Gather all key-value pairs
    std::vector<std::pair<KeyType, ValueType>> all;
    all.reserve(total);

    for (const auto& s : slots_) {
        if (s.isKV()) {
            all.emplace_back(decodeKey(s.key), s.value);
        } else if (s.isPointer()) {
            OverflowBuffer* buffer = s.overflowPtr;
            if (buffer) {
                // Prefer get_all() — data() materializes a lasting view_cache_ copy.
                auto vec = buffer->get_all();
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
    if (!hasSlotLocks()) return false;
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (getSlotMutex(i).try_lock()) {
            getSlotMutex(i).unlock();
        } else {
            return true;
        }
    }
    return false;
}

std::vector<std::unique_lock<HyperSlotMutex>> LeafNode::tryLockAllSlots() {
    std::vector<std::unique_lock<HyperSlotMutex>> locks;
    if (!hasSlotLocks()) return locks;
    locks.reserve(slots_.size());

    for (size_t i = 0; i < slots_.size(); ++i) {
        std::unique_lock<HyperSlotMutex> lock(getSlotMutex(i), std::try_to_lock);
        if (!lock.owns_lock()) {
            locks.clear();
            return locks;
        }
        locks.push_back(std::move(lock));
    }

    return locks;
}

bool LeafNode::tryRetrainInPlace(double delta) {
    auto data = gatherAll();
    if (data.size() < 2) return true;

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
                if (cs.get_last_x() >= maxKey) seg.max_key = data.back().first;
                auto [segSlope, intercept] = cs.get_floating_point_segment(seg.min_key);
                (void)intercept;
                seg.slope = segSlope;
                auto start_it = std::lower_bound(data.begin(), data.end(), seg.min_key,
                                                 [](const auto& pair, KeyType key) { return pair.first < key; });
                seg.start_idx = std::distance(data.begin(), start_it);
                auto end_it = std::lower_bound(data.begin(), data.end(), seg.max_key,
                                               [](const auto& pair, KeyType key) { return pair.first < key; });
                seg.end_idx = std::distance(data.begin(), end_it);
                segments.push_back(seg);
            });
    if (!segments.empty() && segments.back().start_idx >= data.size()) {
        segments.pop_back();
    }

    // Cheap retrain only when a single PLA segment still covers the leaf.
    if (segments.size() != 1) return false;

    const auto& seg = segments.front();
    KeyType saved_max = maxPossibleKey_;
    for (auto& slot : slots_) {
        slot.destroy();
    }
    slope_ = seg.slope;
    minKey_ = seg.min_key;
    MR_ = static_cast<size_t>(std::ceil(slope_ * static_cast<double>(seg.max_key - minKey_)));
    slots_ = std::vector<Slot>(MR_ + 1);
    maxPossibleKey_ = saved_max;
    bulkLoad(std::move(data));
    return maxConflictCount() < kMaxLeafConflicts;
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
            OverflowBuffer *buffer = s.overflowPtr;
            if (buffer) count = buffer->size();
        }
        current_hist[i] = count;
    }

    int n = 0, m = 0;
    if (!init_histogram_ || init_histogram_len_ != current_hist.size()) {
        op_counter_ptr_.store(0, std::memory_order_relaxed);
        op_counter_st_ = 0;
        return false;
    }
    for (size_t i = 0; i < init_histogram_len_; ++i) n += init_histogram_[i];
    for (int x : current_hist) m += x;
    if (n <= 0 || m <= 0 || !init_histogram_) {
        op_counter_ptr_.store(0, std::memory_order_relaxed);
        op_counter_st_ = 0;
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

    op_counter_ptr_.store(0, std::memory_order_relaxed);
    op_counter_st_ = 0;
    return d_stat > threshold;
}
