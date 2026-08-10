/**
 * Regression: bulk load then many inserts that force leaf splits must finish
 * (previously livelocked in ModelInnerNode::setChild seqlock self-spin).
 */
#include "../include/hyper_index.h"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
    Hyper::setLockingEnabled(false);

    Hyper idx;
    std::vector<std::pair<KeyType, ValueType>> data;
    data.reserve(20000);
    for (KeyType k = 0; k < 20000; ++k) {
        data.emplace_back(k * 16 + 1, k);
    }
    idx.bulkLoad(data);

    auto t0 = std::chrono::steady_clock::now();
    for (KeyType k = 0; k < 20000; ++k) {
        idx.insert(k * 16 + 9, k + 100000);
    }
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0)
                  .count();

    // Spot-check a few keys from each wave.
    if (!idx.find(1).has_value() || !idx.find(9).has_value() ||
        !idx.find(16 * 19999 + 9).has_value()) {
        std::cerr << "FAIL: missing keys after growth inserts\n";
        return 1;
    }

    std::cout << "ALL PASS (insert growth)  20k inserts in " << ms << " ms\n";
    auto m = idx.memoryStats();
    std::cout << "  memory total=" << m.total_bytes()
              << " index=" << m.index_bytes << " leaf=" << m.leaf_bytes << "\n";
    Hyper::setLockingEnabled(true);
    return 0;
}
