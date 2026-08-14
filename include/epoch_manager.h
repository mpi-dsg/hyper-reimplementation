#pragma once
#include <atomic>
#include <cassert>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common_defs.h"

#define USE_EPOCHS true

/**
 * @brief Thread-safe epoch-based reclamation for RCU updates.
 *
 * Public operations enter/exit an epoch via Guard. Retired objects are deleted
 * once no active thread remains in their epoch (or earlier).
 *
 * When locking is disabled (single-thread eval), safeDelete deletes immediately
 * so ST runs do not accumulate retired lists.
 */
class EpochManager {
public:
    using Deleter = std::function<void(void*)>;

    static EpochManager& get() {
        static EpochManager instance;
        return instance;
    }

    static int& nestDepth() {
        thread_local int nest = 0;
        return nest;
    }

    void enterEpoch() {
        if (nestDepth()++ > 0) return;
        size_t tid = threadId();
        size_t current_epoch = global_epoch_.load(std::memory_order_acquire);
        thread_epochs_[tid].epoch.store(current_epoch, std::memory_order_release);
        thread_epochs_[tid].active.store(true, std::memory_order_release);
    }

    void exitEpoch() {
        int& nest = nestDepth();
        if (nest <= 0) return;
        if (--nest > 0) return;
        size_t tid = threadId();
        thread_epochs_[tid].active.store(false, std::memory_order_release);
        maybeAdvance();
    }

    void retire(void* ptr, Deleter deleter) {
        if (ptr == nullptr) return;

        thread_local bool in_cleanup = false;
        if (in_cleanup) {
            deleter(ptr);
            return;
        }

        std::lock_guard<std::mutex> lock(retire_mutex_);
        size_t current_epoch = global_epoch_.load(std::memory_order_acquire);
        retired_objects_[current_epoch].emplace_back(ptr, std::move(deleter));
        ++retired_count_;
        if (retired_count_ >= retire_threshold_) {
            advanceEpochInternal();
        }
    }

    void advanceEpoch() {
        std::lock_guard<std::mutex> lock(retire_mutex_);
        advanceEpochInternal();
    }

    void forceReclamation() {
        std::lock_guard<std::mutex> lock(retire_mutex_);
        thread_local bool in_cleanup = true;
        for (auto& [epoch, objects] : retired_objects_) {
            (void)epoch;
            for (auto& [ptr, deleter] : objects) {
                if (ptr != nullptr) deleter(ptr);
            }
        }
        retired_objects_.clear();
        retired_count_ = 0;
        in_cleanup = false;
    }

    class Guard {
    public:
        Guard() { EpochManager::get().enterEpoch(); }
        ~Guard() { EpochManager::get().exitEpoch(); }
    };

    template<typename T>
    void safeDelete(T* ptr) {
        if (ptr == nullptr) return;
        // ST unlocked mode: no concurrent readers of retired objects.
        if (!isHyperLockingEnabled()) {
            delete ptr;
            return;
        }
#if USE_EPOCHS == true
        retire(ptr, [](void* p) { delete static_cast<T*>(p); });
#else
        delete ptr;
#endif
    }

private:
    struct ThreadEpoch {
        std::atomic<size_t> epoch{0};
        std::atomic<bool> active{false};
    };

    size_t threadId() {
        thread_local size_t id = [this]() {
            size_t tid = next_thread_id_.fetch_add(1, std::memory_order_relaxed);
            assert(tid < 64);
            return tid;
        }();
        return id;
    }

    void maybeAdvance() {
        if (retired_count_.load(std::memory_order_relaxed) >= retire_threshold_) {
            advanceEpoch();
        }
    }

    void advanceEpochInternal() {
        global_epoch_.fetch_add(1, std::memory_order_acq_rel);

        size_t min_epoch = global_epoch_.load(std::memory_order_acquire);
        for (size_t i = 0; i < 64; ++i) {
            if (thread_epochs_[i].active.load(std::memory_order_acquire)) {
                min_epoch = std::min(
                    min_epoch,
                    thread_epochs_[i].epoch.load(std::memory_order_acquire));
            }
        }

        thread_local bool in_cleanup = true;
        size_t deleted_count = 0;
        const size_t max_deletions_per_advance = 256;

        for (auto it = retired_objects_.begin();
             it != retired_objects_.end() && deleted_count < max_deletions_per_advance;) {
            if (it->first < min_epoch) {
                for (auto& [ptr, deleter] : it->second) {
                    if (ptr != nullptr) {
                        deleter(ptr);
                        ++deleted_count;
                        if (retired_count_ > 0) --retired_count_;
                    }
                }
                it = retired_objects_.erase(it);
            } else {
                ++it;
            }
        }
        in_cleanup = false;
    }

    std::atomic<size_t> global_epoch_{0};
    std::atomic<size_t> next_thread_id_{0};
    std::atomic<size_t> retired_count_{0};
    ThreadEpoch thread_epochs_[64];
    const size_t retire_threshold_ = 32;

    std::mutex retire_mutex_;
    std::unordered_map<size_t, std::vector<std::pair<void*, Deleter>>> retired_objects_;
};

template<typename T>
void safeDelete(T* ptr) {
    EpochManager::get().safeDelete(ptr);
}

// ST unlocked eval: skip epoch enter/exit atomics on the op path.
#define EPOCH_GUARD()                                                          \
    struct _HyperEpochMaybeGuard {                                             \
        bool active_;                                                          \
        _HyperEpochMaybeGuard() : active_(isHyperLockingEnabled()) {           \
            if (active_) EpochManager::get().enterEpoch();                     \
        }                                                                      \
        ~_HyperEpochMaybeGuard() {                                             \
            if (active_) EpochManager::get().exitEpoch();                      \
        }                                                                      \
    } _epoch_guard

#ifdef EPOCH_GUARD_FORCE
#undef EPOCH_GUARD
#define EPOCH_GUARD() EpochManager::Guard _epoch_guard
#endif
