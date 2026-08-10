/**
 * Smoke tests for paper-style index vs total memory accounting (Fig. 11).
 */
#include "../include/hyper_index.h"

#include <iostream>
#include <vector>

static int failures = 0;

#define EXPECT(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "FAIL: " << msg << "\n";                              \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

int main() {
    Hyper empty;
    auto z = empty.memoryStats();
    EXPECT(z.index_bytes == 0 && z.leaf_bytes == 0, "empty index uses 0 bytes");

    Hyper idx;
    std::vector<std::pair<KeyType, ValueType>> data;
    for (KeyType k = 0; k < 5000; ++k) {
        data.emplace_back(k * 7 + 3, k);
    }
    idx.bulkLoad(data);

    auto m = idx.memoryStats();
    EXPECT(m.total_bytes() > 0, "loaded index reports non-zero total");
    EXPECT(m.leaf_bytes > 0, "leaf_bytes > 0 after bulk load");
    EXPECT(m.index_bytes > 0, "index_bytes > 0 with inner structure");
    EXPECT(m.leaf_bytes > m.index_bytes,
           "leaves dominate memory (paper insight / Fig. 11)");
    EXPECT(idx.memoryBytes() == m.total_bytes(), "memoryBytes matches total");

    if (failures == 0) {
        std::cout << "ALL PASS (paper memory accounting)\n";
        std::cout << "  index_bytes=" << m.index_bytes
                  << " leaf_bytes=" << m.leaf_bytes
                  << " total=" << m.total_bytes() << "\n";
        return 0;
    }
    std::cerr << failures << " failure(s)\n";
    return 1;
}
