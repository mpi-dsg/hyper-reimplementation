/**
 * Runtime adaptation / reclaim / update coverage for paper §3.3 and §4.4.
 */
#include "../include/hyper_index.h"
#include "../include/epoch_manager.h"

#include <cassert>
#include <iostream>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL: " << msg << "\n";                                \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

int main() {
    Hyper::setLockingEnabled(false);

    // Update replaces values without growing the key set.
    {
        Hyper idx;
        std::vector<std::pair<KeyType, ValueType>> data;
        for (KeyType k = 1; k <= 1000; ++k) data.emplace_back(k * 10, k);
        idx.bulkLoad(data);
        CHECK(idx.update(500, 42), "update existing");
        CHECK(idx.find(500).value_or(0) == 42, "updated value");
        CHECK(!idx.update(7, 1), "update missing key");
    }

    // ST overflow inserts reclaim prior buffers (no unbounded growth).
    {
        Hyper idx;
        std::vector<std::pair<KeyType, ValueType>> data = {{100, 1}, {200, 2}, {300, 3}};
        idx.bulkLoad(data);
        // Force many collisions into the same leaf region.
        for (KeyType i = 0; i < 5000; ++i) {
            idx.insert(150 + (i % 3), i + 1000);
        }
        auto m = idx.memoryStats();
        CHECK(m.total_bytes() < 50 * 1024 * 1024, "memory bounded after ST overflow churn");
        CHECK(idx.find(150).has_value(), "keys still findable");
    }

    // Growth inserts finish with 2x rebuild + policies enabled.
    {
        Hyper idx;
        std::vector<std::pair<KeyType, ValueType>> data;
        for (KeyType k = 0; k < 20000; ++k) data.emplace_back(k * 16 + 1, k);
        idx.bulkLoad(data);
        for (KeyType k = 0; k < 20000; ++k) {
            idx.insert(k * 16 + 9, k + 7);
        }
        CHECK(idx.find(9).has_value(), "post-growth find");
        CHECK(idx.memoryStats().total_bytes() > 0, "memory stats live");
        auto s = idx.scan(1, 10);
        CHECK(s.size() == 10, "scan returns requested count");
    }

    Hyper::setLockingEnabled(true);

    if (failures) {
        std::cerr << failures << " checks failed\n";
        return 1;
    }
    std::cout << "ALL PASS (paper runtime)\n";
    return 0;
}
