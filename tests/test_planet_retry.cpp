/**
 * Diagnose Planet-style RetryFromRoot: bulk-load keys, insert until stuck.
 */
#include "../include/hyper_index.h"
#include "../include/leaf_node.h"
#include "../include/model_inner_node.h"
#include "../include/search_inner_node.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

static void dumpPath(Hyper& idx, KeyType key) {
    void* cur = nullptr;
    // Reach into root via find path mirrors insertAttempt
    // Use public find first
    auto v = idx.find(key);
    std::cout << "find(" << key << ") has_value=" << v.has_value();
    if (v) std::cout << " val=" << *v;
    std::cout << "\n";
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/home/GRE_Longitudinal/datasets/planet";
    size_t n_bulk = argc > 2 ? std::stoull(argv[2]) : 2000000;
    size_t n_ins = argc > 3 ? std::stoull(argv[3]) : 500000;

    Hyper::setLockingEnabled(false);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        return 2;
    }

    std::vector<std::pair<KeyType, ValueType>> data;
    data.reserve(n_bulk);
    for (size_t i = 0; i < n_bulk; ++i) {
        KeyType k;
        if (!in.read(reinterpret_cast<char*>(&k), sizeof(k))) break;
        data.emplace_back(k, static_cast<ValueType>(i));
    }
    std::sort(data.begin(), data.end());
    data.erase(std::unique(data.begin(), data.end()), data.end());
    std::cout << "bulk keys=" << data.size() << " min=" << data.front().first
              << " max=" << data.back().first << "\n";

    Hyper idx;
    idx.bulkLoad(data);

    // Insert keys beyond bulk (next keys from file) to force splits / routing stress
    size_t ok = 0;
    for (size_t i = 0; i < n_ins; ++i) {
        KeyType k;
        if (!in.read(reinterpret_cast<char*>(&k), sizeof(k))) break;
        try {
            idx.insert(k, static_cast<ValueType>(n_bulk + i));
            ++ok;
        } catch (const std::exception& e) {
            std::cerr << "FAIL after " << ok << " inserts on key " << k
                      << " err=" << e.what() << "\n";
            dumpPath(idx, k);
            // Probe insertAttempt leaf bounds via find-like walk
            // Manual walk using same public ops is limited; print neighbors via scan of nearby keys
            for (int d = -3; d <= 3; ++d) {
                KeyType pk = k + d;
                auto r = idx.find(pk);
                std::cout << "  find(" << pk << ")=" << (r ? "hit" : "miss") << "\n";
            }
            return 1;
        }
    }
    std::cout << "PASS inserts=" << ok << "\n";
    return 0;
}
