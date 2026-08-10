/**
 * Regression: bulk load then many inserts that force leaf splits must finish
 * (previously livelocked in ModelInnerNode::setChild seqlock self-spin).
 * Also stresses SearchInner→ModelInner conversion (untagged collectAllData crash).
 */
#include "../include/hyper_index.h"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
    Hyper::setLockingEnabled(false);

    Hyper idx;
    std::vector<std::pair<KeyType, ValueType>> data;
    data.reserve(50000);
    for (KeyType k = 0; k < 50000; ++k) {
        data.emplace_back(k * 16 + 1, k);
    }
    idx.bulkLoad(data);

    auto t0 = std::chrono::steady_clock::now();
    // Dense mid-gap inserts to force many splits and search-node conversions.
    for (KeyType k = 0; k < 50000; ++k) {
        idx.insert(k * 16 + 3, k + 100000);
        idx.insert(k * 16 + 5, k + 200000);
        idx.insert(k * 16 + 7, k + 300000);
        idx.insert(k * 16 + 9, k + 400000);
    }
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0)
                  .count();

    // Spot-check a few keys from each wave.
    if (!idx.find(1).has_value() || !idx.find(9).has_value() ||
        !idx.find(16 * 49999 + 9).has_value()) {
        std::cerr << "FAIL: missing keys after growth inserts\n";
        return 1;
    }

    std::cout << "ALL PASS (insert growth)  200k inserts in " << ms << " ms\n";
    auto m = idx.memoryStats();
    std::cout << "  memory total=" << m.total_bytes()
              << " index=" << m.index_bytes << " leaf=" << m.leaf_bytes << "\n";
    Hyper::setLockingEnabled(true);
    return 0;
}
