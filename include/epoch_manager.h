#pragma once
#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <cassert>

#define USE_EPOCHS false

// Minimal, thread-safe epoch-based reclamation for RCU
class EpochManager {
public:
    using Deleter = std::function<void(void*)>;

    // Singleton access
    static EpochManager& get() {
        static EpochManager instance;
        return instance;
    }

    // Enter an epoch (call at the start of every public API or critical section)
    void enterEpoch() {
        thread_local size_t thread_id = [this]() {
            size_t id = next_thread_id_.fetch_add(1);
            assert(id < 64);
            return id;
        }();
        
        size_t current_epoch = global_epoch_.load(std::memory_order_acquire);
        thread_epochs_[thread_id].epoch.store(current_epoch, std::memory_order_release);
        thread_epochs_[thread_id].active.store(true, std::memory_order_release);
    }

    // Exit an epoch (call at the end of every public API or critical section)
    void exitEpoch() {
        thread_local size_t thread_id = [this]() {
            // Find our thread ID by scanning backwards
            size_t max_id = next_thread_id_.load();
            for (size_t i = 0; i < max_id && i < 64; ++i) {
                if (thread_epochs_[i].active.load()) {
                    return i;
                }
            }
            return size_t(0);
        }();
        
        thread_epochs_[thread_id].active.store(false, std::memory_order_release);
    }

    // Retire a pointer with a deleter (call when logically removing an object)
    void retire(void* ptr, Deleter deleter) {
        if (ptr == nullptr) return;
        
        // Prevent recursive retirement during cleanup
        thread_local bool in_cleanup = false;
        if (in_cleanup) {
            // Direct deletion during cleanup to avoid infinite recursion
            deleter(ptr);
            return;
        }
        
        std::lock_guard<std::mutex> lock(retire_mutex_);
        size_t current_epoch = global_epoch_.load(std::memory_order_acquire);
        retired_objects_[current_epoch].emplace_back(ptr, std::move(deleter));
        
        // Trigger cleanup if we have too many retired objects
        // if (retired_objects_[current_epoch].size() > retire_threshold_) {
        //     advanceEpochInternal();
        // }
    }

    // Advance the global epoch (should be called periodically, e.g., by a background thread or after N operations)
    void advanceEpoch() {
        std::lock_guard<std::mutex> lock(retire_mutex_);
        advanceEpochInternal();
    }

    // Force reclamation of all retired objects
    void forceReclamation() {
        std::lock_guard<std::mutex> lock(retire_mutex_);
        
        // Set cleanup flag to prevent recursive retirement
        thread_local bool in_cleanup = true;
        
        for (auto& [epoch, objects] : retired_objects_) {
            for (auto& [ptr, deleter] : objects) {
                if (ptr != nullptr) {
                    deleter(ptr);
                }
            }
        }
        retired_objects_.clear();
        
        in_cleanup = false;
    }

    // RAII guard for automatic epoch entry/exit
    class Guard {
    public:
        Guard() { EpochManager::get().enterEpoch(); }
        ~Guard() { EpochManager::get().exitEpoch(); }
    };

    // Convenience function for safe deletion with default deleter
    template<typename T>
    void safeDelete(T* ptr) {
#if USE_EPOCHS == true
        if (ptr != nullptr) {
            retire(ptr, [](void* p) { delete static_cast<T*>(p); });
        }
#endif
    }

private:
    struct ThreadEpoch {
        std::atomic<size_t> epoch{0};
        std::atomic<bool> active{false};
    };

    void advanceEpochInternal() {
        size_t old_epoch = global_epoch_.fetch_add(1, std::memory_order_acq_rel);
        
        // Find the minimum epoch among all active threads
        size_t min_epoch = global_epoch_.load();
        for (size_t i = 0; i < 64; ++i) {
            if (thread_epochs_[i].active.load(std::memory_order_acquire)) {
                size_t thread_epoch = thread_epochs_[i].epoch.load(std::memory_order_acquire);
                min_epoch = std::min(min_epoch, thread_epoch);
            }
        }
        
        // Set cleanup flag to prevent recursive retirement
        thread_local bool in_cleanup = true;
        
        // Delete objects from epochs that are now safe
        // Process in batches to avoid too much work in one go
        size_t deleted_count = 0;
        const size_t max_deletions_per_advance = 100;
        
        for (auto it = retired_objects_.begin(); it != retired_objects_.end() && deleted_count < max_deletions_per_advance;) {
            if (it->first < min_epoch) {
                for (auto& [ptr, deleter] : it->second) {
                    if (ptr != nullptr) {
                        deleter(ptr);
                        deleted_count++;
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
    ThreadEpoch thread_epochs_[64];
    const size_t retire_threshold_ = 32;
    
    std::mutex retire_mutex_;
    std::unordered_map<size_t, std::vector<std::pair<void*, Deleter>>> retired_objects_;
};

/**
 * @brief Helper function to schedule a pointer for safe deletion
 * @tparam T Type of the object to delete
 * @param ptr Pointer to delete
 */
template<typename T>
void safeDelete(T* ptr) {
    EpochManager::get().safeDelete(ptr);
}

/**
 * @brief Helper macro to create an epoch guard
 */
#if USE_EPOCHS == true
#define EPOCH_GUARD() EpochManager::Guard _epoch_guard
#else
#define EPOCH_GUARD() (void)0
#endif