/**
 * Smoke tests for Hyper paper §4.4 delete semantics.
 */
#include "../include/hyper_index.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

static int failures = 0;

#define EXPECT(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "FAIL: " << msg << "\n";                              \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

static void test_erase_basic() {
    Hyper idx;
    std::vector<std::pair<KeyType, ValueType>> data;
    for (KeyType k = 1; k <= 1000; ++k) {
        data.emplace_back(k, k * 10);
    }
    idx.bulkLoad(data);

    EXPECT(idx.find(500).has_value(), "key 500 present before erase");
    EXPECT(idx.erase(500), "erase(500) returns true");
    EXPECT(!idx.find(500).has_value(), "key 500 gone after erase");
    EXPECT(!idx.erase(500), "second erase(500) returns false");
    EXPECT(idx.find(499).value_or(0) == 4990, "neighbor 499 intact");
    EXPECT(idx.find(501).value_or(0) == 5010, "neighbor 501 intact");
}

static void test_erase_leftmost_metadata() {
    // Bulk-load a contiguous range so the first key becomes a leaf minKey.
    Hyper idx;
    std::vector<std::pair<KeyType, ValueType>> data;
    for (KeyType k = 100; k < 300; ++k) {
        data.emplace_back(k, k);
    }
    idx.bulkLoad(data);

    EXPECT(idx.erase(100), "erase leftmost key of dataset");
    EXPECT(!idx.find(100).has_value(), "leftmost key value removed");
    // Remaining keys in the same region must still be findable — minKey_
    // metadata was left in place so the leaf still covers the range.
    EXPECT(idx.find(101).has_value(), "key 101 still reachable after leftmost erase");
    EXPECT(idx.find(150).has_value(), "key 150 still reachable after leftmost erase");
}

static void test_erase_then_reinsert() {
    Hyper idx;
    std::vector<std::pair<KeyType, ValueType>> data;
    for (KeyType k = 0; k < 500; ++k) {
        data.emplace_back(k * 3 + 1, k);
    }
    idx.bulkLoad(data);

    EXPECT(idx.erase(1), "erase key 1");
    idx.insert(1, 999);
    auto v = idx.find(1);
    EXPECT(v.has_value() && *v == 999, "reinsert after erase works");
}

int main() {
    test_erase_basic();
    test_erase_leftmost_metadata();
    test_erase_then_reinsert();

    if (failures == 0) {
        std::cout << "ALL PASS (paper §4.4 delete)\n";
        return 0;
    }
    std::cerr << failures << " failure(s)\n";
    return 1;
}
