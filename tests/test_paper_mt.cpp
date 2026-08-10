/**
 * Lightweight multi-thread smoke: concurrent finds/inserts with locking+epochs.
 */
#include "../include/hyper_index.h"

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    Hyper::setLockingEnabled(true);

    Hyper idx;
    std::vector<std::pair<KeyType, ValueType>> data;
    data.reserve(20000);
    for (KeyType k = 0; k < 20000; ++k) {
        data.emplace_back(k * 10 + 1, k);
    }
    idx.bulkLoad(data);

    std::atomic<int> errors{0};
    constexpr int kThreads = 4;
    constexpr int kOps = 5000;

    auto worker = [&](int tid) {
        for (int i = 0; i < kOps; ++i) {
            KeyType base = static_cast<KeyType>((tid * kOps + i) % 20000);
            KeyType existing = base * 10 + 1;
            if (!idx.find(existing).has_value()) {
                errors.fetch_add(1);
            }
            KeyType neu = base * 10 + 3;
            try {
                idx.insert(neu, static_cast<ValueType>(tid + i));
            } catch (...) {
                errors.fetch_add(1);
            }
            auto v = idx.find(neu);
            if (!v.has_value()) {
                errors.fetch_add(1);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) th.join();

    Hyper::setLockingEnabled(false);

    if (errors.load() != 0) {
        std::cerr << "FAIL: mt errors=" << errors.load() << "\n";
        return 1;
    }
    std::cout << "ALL PASS (paper mt smoke)\n";
    return 0;
}
