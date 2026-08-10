/**
 * Paper §6.1: single-thread mode disables Hyper locking.
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
    // Ensure clean default, then exercise ST path.
    Hyper::setLockingEnabled(true);
    EXPECT(Hyper::lockingEnabled(), "locking enabled by default");

    Hyper::setLockingEnabled(false);
    EXPECT(!Hyper::lockingEnabled(), "setLockingEnabled(false) sticks");

    Hyper idx;
    std::vector<std::pair<KeyType, ValueType>> data;
    for (KeyType k = 1; k <= 200; ++k) data.emplace_back(k, k);
    idx.bulkLoad(data);

    idx.insert(201, 201);
    EXPECT(idx.find(201).value_or(0) == 201, "insert works with locks off");
    EXPECT(idx.erase(50), "erase works with locks off");
    EXPECT(!idx.find(50).has_value(), "erased key gone with locks off");

    // Restore default for other tests in the same process / subsequent runs.
    Hyper::setLockingEnabled(true);
    EXPECT(Hyper::lockingEnabled(), "restored locking default");

    if (failures == 0) {
        std::cout << "ALL PASS (paper §6.1 ST lock disable)\n";
        return 0;
    }
    std::cerr << failures << " failure(s)\n";
    return 1;
}
