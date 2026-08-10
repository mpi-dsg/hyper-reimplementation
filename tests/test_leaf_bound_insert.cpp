/**
 * Regression: keys equal to inclusive maxPossibleKey must insert, not RetryFromRoot.
 */
#include "../include/hyper_index.h"
#include "../include/leaf_node.h"

#include <iostream>
#include <limits>
#include <vector>

int main() {
    Hyper::setLockingEnabled(false);

    // Contiguous keys so PLA splits create adjacent leaves with max = next.min-1.
    std::vector<std::pair<KeyType, ValueType>> data;
    data.reserve(5000);
    for (KeyType k = 1; k <= 5000; ++k) data.emplace_back(k, k);

    Hyper idx;
    idx.bulkLoad(data);

    // Force many inserts in dense gaps / boundaries.
    for (KeyType k = 5001; k <= 20000; ++k) {
        try {
            idx.insert(k, k);
        } catch (const std::exception& e) {
            std::cerr << "FAIL insert key=" << k << " " << e.what() << "\n";
            return 1;
        }
        if (!idx.find(k).has_value()) {
            std::cerr << "FAIL find miss key=" << k << "\n";
            return 1;
        }
    }

    // Direct leaf bound check
    LeafNode leaf(1.0, 100, 200);
    leaf.setMaxPossibleKey(199);
    leaf.bulkLoad({{100, 1}, {150, 2}, {199, 3}});
    auto r = leaf.insert(199, 9, 128.0);
    if (r.result == InsertResult::RetryFromRoot) {
        std::cerr << "FAIL: insert at inclusive maxPossibleKey retried\n";
        return 1;
    }

    std::cout << "ALL PASS (leaf bound insert)\n";
    return 0;
}
