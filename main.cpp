#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <functional>
#include <map>
#include <cmath>
#include <fstream>
#include <unordered_set>
#include "include/hyper_index.h"
#include <atomic>
#include <cassert>
#include <cstdint>
#include <omp.h>

using namespace std;
using namespace std::chrono;

using KVPair = pair<uint64_t, uint64_t>;

// Global thread count configuration
int NUM_THREADS = 8;

// --- Synthetic Data Generation Functions ---

// Generate sorted key–value pairs uniformly over [min, max].
vector<KVPair> generateDataUniform(size_t num, uint64_t min = 0, uint64_t max = (1ULL << 63) - 1) {
    vector<KVPair> data;
    data.reserve(num);
    mt19937_64 gen(random_device{}());
    if (min >= max) {
        max = min + 1;
        if (max == 0) { min = 0; max = UINT64_MAX; }
    }
    uniform_int_distribution<uint64_t> dist(min, max);
    for (size_t i = 0; i < num; i++) {
        uint64_t key = dist(gen);
        data.emplace_back(key, key);
    }
    sort(data.begin(), data.end(), [](const KVPair &a, const KVPair &b) {
        return a.first < b.first;
    });
    return data;
}

vector<KVPair> generateDataUniformParallel(size_t num, uint64_t min = 0, uint64_t max = (1ULL << 63) - 1) {
    vector<KVPair> data(num);
    if (min >= max) {
        max = min + 1;
        if (max == 0) { min = 0; max = UINT64_MAX; }
    }

#pragma omp parallel
    {
        thread_local mt19937_64 gen(random_device{}() + omp_get_thread_num());
        uniform_int_distribution<uint64_t> dist(min, max);

#pragma omp for
        for (size_t i = 0; i < num; i++) {
            uint64_t key = dist(gen);
            data[i] = {key, key};
        }
    }

    sort(data.begin(), data.end(), [](const KVPair &a, const KVPair &b) {
        return a.first < b.first;
    });
    return data;
}

vector<uint64_t> generateQueryKeys(const vector<KVPair> &bulkData, size_t n) {
    vector<uint64_t> queries;
    queries.reserve(n);

    if (bulkData.empty()) return queries;

    mt19937_64 gen(random_device{}());
    uniform_int_distribution<size_t> dist(0, bulkData.size() - 1);

    for (size_t i = 0; i < n; ++i) {
        size_t idx = dist(gen);
        queries.push_back(bulkData[idx].first);
    }
    return queries;
}

vector<uint64_t> generateQueryKeysParallel(const vector<KVPair> &bulkData, size_t n) {
    vector<uint64_t> queries(n);
    if (bulkData.empty()) return queries;

#pragma omp parallel
    {
        thread_local mt19937_64 gen(random_device{}() + omp_get_thread_num());
        uniform_int_distribution<size_t> dist(0, bulkData.size() - 1);

#pragma omp for
        for (size_t i = 0; i < n; ++i) {
            size_t idx = dist(gen);
            queries[i] = bulkData[idx].first;
        }
    }
    return queries;
}

// --- File-Based Data Loading Functions ---

vector<KVPair> loadBulkData(const string &file_path, size_t n) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_path << endl;
        return {};
    }

    // Determine total number of keys in the file.
    file.seekg(0, ios::end);
    size_t fileSize = file.tellg();
    size_t total_keys = fileSize / sizeof(uint64_t);
    if (total_keys <= 1) {
        cerr << "File " << file_path << " does not contain enough keys." << endl;
        return {};
    }
    size_t available_keys = total_keys - 1; // Ignore the first dummy key.

    // Load the entire dataset (except dummy) into memory.
    vector<uint64_t> keys(available_keys);
    file.seekg(sizeof(uint64_t), ios::beg); // Skip dummy key.
    file.read(reinterpret_cast<char*>(keys.data()), available_keys * sizeof(uint64_t));
    // Keys are already sorted as the dataset is sorted.

    vector<KVPair> bulk;
    if (n >= available_keys) {
        // Use the entire dataset.
        bulk.reserve(available_keys);
        for (auto key : keys)
            bulk.push_back({key, key});
    } else {
        // Sample n distinct indices uniformly from [0, available_keys - 1].
        unordered_set<size_t> indexSet;
        mt19937_64 gen(random_device{}());
        uniform_int_distribution<size_t> dist(0, available_keys - 1);
        while (indexSet.size() < n) {
            indexSet.insert(dist(gen));
        }
        vector<size_t> indices(indexSet.begin(), indexSet.end());
        sort(indices.begin(), indices.end());
        bulk.reserve(n);
        for (size_t idx : indices) {
            bulk.push_back({keys[idx], keys[idx]});
        }
    }
    // Remove duplicates (if any exist in the dataset).
    bulk.erase(unique(bulk.begin(), bulk.end(), [](const KVPair &a, const KVPair &b) {
        return a.first == b.first;
    }), bulk.end());

    return bulk;
}

vector<uint64_t> loadRandomFindQueryKeys(const string &file_path, size_t n) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_path << endl;
        return {};
    }

    // Determine total number of keys in the file.
    file.seekg(0, ios::end);
    size_t fileSize = file.tellg();
    size_t total_keys = fileSize / sizeof(uint64_t);
    if (total_keys <= 1) {
        cerr << "File " << file_path << " does not contain enough keys." << endl;
        return {};
    }
    size_t available_keys = total_keys - 1; // Ignore dummy key.

    // Load the entire dataset (except dummy) into memory.
    vector<uint64_t> keys(available_keys);
    file.seekg(sizeof(uint64_t), ios::beg); // Skip dummy key.
    file.read(reinterpret_cast<char*>(keys.data()), available_keys * sizeof(uint64_t));

    // Sample n keys uniformly with replacement (duplicates allowed).
    vector<uint64_t> queries;
    queries.reserve(n);
    mt19937_64 gen(random_device{}());
    uniform_int_distribution<size_t> dist(0, available_keys - 1);
    for (size_t i = 0; i < n; i++) {
        queries.push_back(keys[dist(gen)]);
    }
    return queries;
}

// --- Multithreaded Benchmark Workload Functions ---

double runReadOnlyBenchmark(const vector<KVPair> &bulkData, size_t totalQueries) {
    Hyper idx;
    idx.bulkLoad(bulkData);

    // Teile die Gesamtanzahl der Queries auf die Threads auf
    size_t queriesPerThread = totalQueries / NUM_THREADS;
    size_t remainingQueries = totalQueries % NUM_THREADS;

    vector<vector<uint64_t>> all_thread_queries(NUM_THREADS);

#pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        thread_local mt19937_64 gen(random_device{}() + thread_id);

        // Jeder Thread bekommt queriesPerThread Queries, der erste Thread bekommt zusätzlich remainingQueries
        size_t threadQueries = queriesPerThread;
        if (thread_id == 0) {
            threadQueries += remainingQueries;
        }

        if (!bulkData.empty()) {
            uniform_int_distribution<size_t> dist(0, bulkData.size() - 1);
            all_thread_queries[thread_id].reserve(threadQueries);
            for (size_t i = 0; i < threadQueries; ++i) {
                size_t idx = dist(gen);
                all_thread_queries[thread_id].push_back(bulkData[idx].first);
            }
        } else {
            all_thread_queries[thread_id].resize(threadQueries, 0); // Fallback with dummy values
        }
    }

    size_t total_operations = 0;

    double start_time = omp_get_wtime();

#pragma omp parallel reduction(+: total_operations) num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        const auto& thread_queries = all_thread_queries[thread_id];

        for (auto key : thread_queries) {
            auto res = idx.find(key);
            if (key != res) {
                // Prevent optimization
                volatile auto dummy = res;
                (void)dummy;
            }
            total_operations++;
        }
    }

    double end_time = omp_get_wtime();
    double durationSec = end_time - start_time;
    return (durationSec > 0) ? (total_operations / durationSec / 1e6) : 0;
}

double runWriteOnlyBenchmark(const vector<KVPair> &bulkData, size_t totalInserts) {
    Hyper idx;
    idx.bulkLoad(bulkData);

    // Teile die Gesamtanzahl der Inserts auf die Threads auf
    size_t insertsPerThread = totalInserts / NUM_THREADS;
    size_t remainingInserts = totalInserts % NUM_THREADS;

    vector<vector<KVPair>> all_thread_inserts(NUM_THREADS);

#pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        thread_local mt19937_64 gen(random_device{}() + thread_id);

        // Jeder Thread bekommt insertsPerThread Inserts, der erste Thread bekommt zusätzlich remainingInserts
        size_t threadInserts = insertsPerThread;
        if (thread_id == 0) {
            threadInserts += remainingInserts;
        }

        uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
        all_thread_inserts[thread_id].reserve(threadInserts);

        for (size_t i = 0; i < threadInserts; ++i) {
            uint64_t key = dist(gen);
            all_thread_inserts[thread_id].emplace_back(key, key);
        }

        shuffle(all_thread_inserts[thread_id].begin(), all_thread_inserts[thread_id].end(), gen);
    }

    size_t total_operations = 0;

    double start_time = omp_get_wtime();

#pragma omp parallel reduction(+: total_operations) num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        const auto& thread_inserts = all_thread_inserts[thread_id];

        for (const auto &kv : thread_inserts) {
            idx.insert(kv.first, kv.second);
            total_operations++;
        }
    }

    double end_time = omp_get_wtime();
    double durationSec = end_time - start_time;
    return (durationSec > 0) ? (total_operations / durationSec / 1e6) : 0;
}

enum class OpType { Insert, Query };
struct Operation {
    OpType type;
    uint64_t key;
    uint64_t value; // For insert operations.
};

double runReadWriteBenchmark(const vector<KVPair> &bulkData, size_t totalInserts, size_t totalQueries) {
    Hyper idx;
    idx.bulkLoad(bulkData);

    size_t n_ops = min(totalInserts, totalQueries);
    if (n_ops == 0) return 0;

    // Teile die Gesamtanzahl der Operationen auf die Threads auf
    size_t opsPerThread = n_ops / NUM_THREADS;
    size_t remainingOps = n_ops % NUM_THREADS;

    vector<vector<Operation>> all_thread_ops(NUM_THREADS);

#pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        thread_local mt19937_64 gen(random_device{}() + thread_id);

        // Jeder Thread bekommt opsPerThread Operationen, der erste Thread bekommt zusätzlich remainingOps
        size_t threadOps = opsPerThread;
        if (thread_id == 0) {
            threadOps += remainingOps;
        }

        all_thread_ops[thread_id].reserve(2 * threadOps);

        uniform_int_distribution<uint64_t> insert_dist(0, UINT64_MAX);

        vector<uint64_t> thread_queries;
        if (!bulkData.empty()) {
            uniform_int_distribution<size_t> query_dist(0, bulkData.size() - 1);
            for (size_t i = 0; i < threadOps; ++i) {
                thread_queries.push_back(bulkData[query_dist(gen)].first);
            }
        } else {
            thread_queries.resize(threadOps, 0);
        }

        for (size_t i = 0; i < threadOps; i++) {
            uint64_t insert_key = insert_dist(gen);
            all_thread_ops[thread_id].push_back({OpType::Insert, insert_key, insert_key});
            all_thread_ops[thread_id].push_back({OpType::Query, thread_queries[i], 0});
        }

        shuffle(all_thread_ops[thread_id].begin(), all_thread_ops[thread_id].end(), gen);
    }

    size_t total_operations = 0;

    double start_time = omp_get_wtime();

#pragma omp parallel reduction(+: total_operations) num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        const auto& thread_ops = all_thread_ops[thread_id];

        for (const auto &op : thread_ops) {
            if (op.type == OpType::Insert) {
                idx.insert(op.key, op.value);
            } else {
                auto res = idx.find(op.key);
                if (op.key != res) {
                    volatile auto dummy = res;
                    (void)dummy;
                }
            }
            total_operations++;
        }
    }

    double end_time = omp_get_wtime();
    double durationSec = end_time - start_time;
    return (durationSec > 0) ? (total_operations / durationSec / 1e6) : 0;
}

double runBulkLoadBenchmark(const vector<KVPair> &bulkData) {
    // Für Bulk Load macht es keinen Sinn, die Daten auf Threads aufzuteilen,
    // da jeder Thread eine separate Instanz erstellt.
    // Hier bleibt die ursprüngliche Implementierung bestehen.
    size_t total_operations = 0;

    double start_time = omp_get_wtime();

#pragma omp parallel reduction(+: total_operations) num_threads(NUM_THREADS)
    {
        Hyper idx;
        idx.bulkLoad(bulkData);
        total_operations += bulkData.size();
    }

    double end_time = omp_get_wtime();
    double durationSec = end_time - start_time;
    return (durationSec > 0 && !bulkData.empty()) ? (total_operations / durationSec / 1e6) : 0;
}

double runRangeQueryBenchmark(const vector<KVPair> &bulkData, size_t totalRangeQueries) {
    Hyper idx;
    idx.bulkLoad(bulkData);
    if (bulkData.empty()) return 0;

    // Teile die Gesamtanzahl der Range Queries auf die Threads auf
    size_t rangeQueriesPerThread = totalRangeQueries / NUM_THREADS;
    size_t remainingRangeQueries = totalRangeQueries % NUM_THREADS;

    vector<vector<pair<uint64_t, uint64_t>>> all_thread_ranges(NUM_THREADS);

#pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        thread_local mt19937_64 gen(random_device{}() + thread_id);
        uniform_int_distribution<size_t> start_index_dist(0, bulkData.size() - 1);
        uniform_int_distribution<size_t> length_dist(1, 99);

        // Jeder Thread bekommt rangeQueriesPerThread Range Queries, der erste Thread bekommt zusätzlich remainingRangeQueries
        size_t threadRangeQueries = rangeQueriesPerThread;
        if (thread_id == 0) {
            threadRangeQueries += remainingRangeQueries;
        }

        all_thread_ranges[thread_id].reserve(threadRangeQueries);

        for (size_t i = 0; i < threadRangeQueries; ++i) {
            size_t start_idx = start_index_dist(gen);
            size_t length = length_dist(gen);
            size_t end_idx = min(bulkData.size() - 1, start_idx + length - 1);
            uint64_t start_key = bulkData[start_idx].first;
            uint64_t end_key = bulkData[end_idx].first;
            if (start_key > end_key)
                swap(start_key, end_key);
            all_thread_ranges[thread_id].push_back({start_key, end_key});
        }
    }

    size_t total_operations = 0;

    double start_time = omp_get_wtime();

#pragma omp parallel reduction(+: total_operations) num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        const auto& thread_ranges = all_thread_ranges[thread_id];

        for (const auto &range : thread_ranges) {
            idx.rangeQuery(range.first, range.second);
            total_operations++;
        }
    }

    double end_time = omp_get_wtime();
    double durationSec = end_time - start_time;
    return (durationSec > 0) ? (total_operations / durationSec) : 0;
}

// --- Main Function with Multiple Iterations and CSV Logging ---
int main() {
    // Set number of threads (can be configured)
    cout << "Enter number of threads (default 8): ";
    string input;
    getline(cin, input);
    if (!input.empty()) {
        NUM_THREADS = stoi(input);
    }

    cout << "Running benchmarks with " << NUM_THREADS << " threads" << endl;
    cout << "Note: Total workload will be divided among threads" << endl;
    omp_set_num_threads(NUM_THREADS);

    // Number of iterations per workload.
    const int iterations = 3;

    // Open CSV file for logging results.
    ofstream logFile("benchmark_results_mt.csv");
    if (!logFile.is_open()) {
        cerr << "Error opening benchmark_results_mt.csv for writing." << endl;
        return 1;
    }
    // Write CSV header.
    logFile << "Workload,Dataset,Iteration,Threads,Throughput" << endl;

    // Define sizes for synthetic benchmarks
    size_t numBulk200 = 200'000'000;
    size_t numBulk100 = 100'000'000;
    size_t numQueries = 400'000'000;
    size_t numInserts = 100'000'000;
    size_t numRangeQueries = 1'000'000;

    // Maximum key value for synthetic data
    uint64_t max_key_value = std::numeric_limits<uint64_t>::max();

    cout << "\n--- Multithreaded Synthetic Benchmarks (Uniform Data) ---" << endl;
    cout << "Total operations will be divided among " << NUM_THREADS << " threads" << endl;

    // Read-Only Benchmark
    double sumRO = 0;
    for (int i = 0; i < iterations; i++) {
        cout << "Generating data for Read-Only benchmark iteration " << i+1 << "..." << endl;
        auto bulkData = generateDataUniformParallel(numBulk200, 0, max_key_value);
        cout << "Starting Read-Only benchmark (" << numQueries << " queries total, ~" << numQueries/NUM_THREADS << " per thread)..." << endl;
        double t = runReadOnlyBenchmark(bulkData, numQueries);
        sumRO += t;
        cout << "[Synthetic Read-Only] Iteration " << i+1 << ": " << t << " million op/sec" << endl;
        logFile << "Synthetic_ReadOnly,Synthetic," << i+1 << "," << NUM_THREADS << "," << t << endl;
    }
    cout << "[Synthetic Read-Only] Average Throughput: " << (sumRO/iterations) << " million op/sec" << endl;

    // Write-Only Benchmark
    double sumWO = 0;
    for (int i = 0; i < iterations; i++) {
        cout << "Generating data for Write-Only benchmark iteration " << i+1 << "..." << endl;
        auto bulkData = generateDataUniformParallel(numBulk100, 0, max_key_value);
        cout << "Starting Write-Only benchmark (" << numInserts << " inserts total, ~" << numInserts/NUM_THREADS << " per thread)..." << endl;
        double t = runWriteOnlyBenchmark(bulkData, numInserts);
        sumWO += t;
        cout << "[Synthetic Write-Only] Iteration " << i+1 << ": " << t << " million op/sec" << endl;
        logFile << "Synthetic_WriteOnly,Synthetic," << i+1 << "," << NUM_THREADS << "," << t << endl;
    }
    cout << "[Synthetic Write-Only] Average Throughput: " << (sumWO/iterations) << " million op/sec" << endl;

    // Read-Write Mixed Benchmark
    double sumRW = 0;
    for (int i = 0; i < iterations; i++) {
        cout << "Generating data for Read-Write Mixed benchmark iteration " << i+1 << "..." << endl;
        auto bulkData = generateDataUniformParallel(numBulk100, 0, max_key_value);
        cout << "Starting Read-Write Mixed benchmark (" << numInserts << " ops total, ~" << numInserts/NUM_THREADS << " per thread)..." << endl;
        double t = runReadWriteBenchmark(bulkData, numInserts, numInserts);
        sumRW += t;
        cout << "[Synthetic Read-Write Mixed] Iteration " << i+1 << ": " << t << " million op/sec" << endl;
        logFile << "Synthetic_ReadWriteMixed,Synthetic," << i+1 << "," << NUM_THREADS << "," << t << endl;
    }
    cout << "[Synthetic Read-Write Mixed] Average Throughput: " << (sumRW/iterations) << " million op/sec" << endl;
    /*
    // Bulk Load Benchmark (bleibt unverändert, da jeder Thread eine separate Instanz erstellt)
    double sumBL = 0;
    for (int i = 0; i < iterations; i++) {
        cout << "Generating data for Bulk Load benchmark iteration " << i+1 << "..." << endl;
        auto bulkData = generateDataUniformParallel(numBulk200, 0, max_key_value);
        cout << "Starting Bulk Load benchmark (each thread loads " << numBulk200 << " items)..." << endl;
        double t = runBulkLoadBenchmark(bulkData);
        sumBL += t;
        cout << "[Synthetic Bulk Load] Iteration " << i+1 << ": " << t << " million op/sec" << endl;
        logFile << "Synthetic_BulkLoad,Synthetic," << i+1 << "," << NUM_THREADS << "," << t << endl;
    }
    cout << "[Synthetic Bulk Load] Average Throughput: " << (sumBL/iterations) << " million op/sec" << endl;

    // Range Query Benchmark
    double sumRQ = 0;
    for (int i = 0; i < iterations; i++) {
        cout << "Generating data for Range Query benchmark iteration " << i+1 << "..." << endl;
        auto bulkData = generateDataUniformParallel(numBulk200, 0, max_key_value);
        cout << "Starting Range Query benchmark (" << numRangeQueries << " queries total, ~" << numRangeQueries/NUM_THREADS << " per thread)..." << endl;
        double t = runRangeQueryBenchmark(bulkData, numRangeQueries);
        sumRQ += t;
        cout << "[Synthetic Range Query] Iteration " << i+1 << ": " << t << " queries/sec" << endl;
        logFile << "Synthetic_RangeQuery,Synthetic," << i+1 << "," << NUM_THREADS << "," << t << endl;
    }
    cout << "[Synthetic Range Query] Average Throughput: " << (sumRQ/iterations) << " queries/sec" << endl;
    */
    // --- Dataset-Based Benchmarks ---
    // Define file paths for your datasets.
    string path1 = "data/fb_200M_uint64";
    string path2 = "data/books_800M_uint64";
    string path3 = "data/osm_cellids_800M_uint64";
    string path4 = "data/wiki_ts_200M_uint64";

    // For dataset benchmarks, use the file-based loading functions.
    auto runDatasetBenchmarks = [&](const string& name, const string &filePath) {
        cout << "\n--- Multithreaded Dataset Benchmark: " << name << " ---" << endl;

        double ds_sumRO = 0;
        for (int i = 0; i < iterations; i++) {
            cout << "Loading data for " << name << " Read-Only benchmark iteration " << i+1 << "..." << endl;
            auto bulkBig = loadBulkData(filePath, numBulk200);
            cout << "Starting Read-Only benchmark (" << numQueries << " queries total, ~" << numQueries/NUM_THREADS << " per thread)..." << endl;
            double t = runReadOnlyBenchmark(bulkBig, numQueries);
            ds_sumRO += t;
            cout << "[Dataset " << name << " - Read-Only] Iteration " << i+1 << ": " << t << " million op/sec" << endl;
            logFile << "Dataset_ReadOnly," << name << "," << i+1 << "," << NUM_THREADS << "," << t << endl;
        }
        cout << "[Dataset " << name << " - Read-Only] Average Throughput: " << (ds_sumRO/iterations) << " million op/sec" << endl;

        double ds_sumWO = 0;
        for (int i = 0; i < iterations; i++) {
            cout << "Loading data for " << name << " Write-Only benchmark iteration " << i+1 << "..." << endl;
            auto bulkSmall = loadBulkData(filePath, numBulk100);
            cout << "Starting Write-Only benchmark (" << numInserts << " inserts total, ~" << numInserts/NUM_THREADS << " per thread)..." << endl;
            double t = runWriteOnlyBenchmark(bulkSmall, numInserts);
            ds_sumWO += t;
            cout << "[Dataset " << name << " - Write-Only] Iteration " << i+1 << ": " << t << " million op/sec" << endl;
            logFile << "Dataset_WriteOnly," << name << "," << i+1 << "," << NUM_THREADS << "," << t << endl;
        }
        cout << "[Dataset " << name << " - Write-Only] Average Throughput: " << (ds_sumWO/iterations) << " million op/sec" << endl;

        double ds_sumRW = 0;
        for (int i = 0; i < iterations; i++) {
            cout << "Loading data for " << name << " Read-Write Mixed benchmark iteration " << i+1 << "..." << endl;
            auto bulkSmall = loadBulkData(filePath, numBulk100);
            cout << "Starting Read-Write Mixed benchmark (" << numInserts << " ops total, ~" << numInserts/NUM_THREADS << " per thread)..." << endl;
            double t = runReadWriteBenchmark(bulkSmall, numInserts, numInserts);
            ds_sumRW += t;
            cout << "[Dataset " << name << " - Read-Write Mixed] Iteration " << i+1 << ": " << t << " million op/sec" << endl;
            logFile << "Dataset_ReadWriteMixed," << name << "," << i+1 << "," << NUM_THREADS << "," << t << endl;
        }
        cout << "[Dataset " << name << " - Read-Write Mixed] Average Throughput: " << (ds_sumRW/iterations) << " million op/sec" << endl;
        /*
        double ds_sumBL = 0;
        for (int i = 0; i < iterations; i++) {
            cout << "Loading data for " << name << " Bulk Load benchmark iteration " << i+1 << "..." << endl;
            auto bulkBig = loadBulkData(filePath, numBulk200);
            cout << "Starting Bulk Load benchmark (each thread loads " << numBulk200 << " items)..." << endl;
            double t = runBulkLoadBenchmark(bulkBig);
            ds_sumBL += t;
            cout << "[Dataset " << name << " - Bulk Load] Iteration " << i+1 << ": " << t << " million op/sec" << endl;
            logFile << "Dataset_BulkLoad," << name << "," << i+1 << "," << NUM_THREADS << "," << t << endl;
        }
        cout << "[Dataset " << name << " - Bulk Load] Average Throughput: " << (ds_sumBL/iterations) << " million op/sec" << endl;

        double ds_sumRQ = 0;
        for (int i = 0; i < iterations; i++) {
            cout << "Loading data for " << name << " Range Query benchmark iteration " << i+1 << "..." << endl;
            auto bulkBig = loadBulkData(filePath, numBulk200);
            cout << "Starting Range Query benchmark (" << numRangeQueries << " queries total, ~" << numRangeQueries/NUM_THREADS << " per thread)..." << endl;
            double t = runRangeQueryBenchmark(bulkBig, numRangeQueries);
            ds_sumRQ += t;
            cout << "[Dataset " << name << " - Range Query] Iteration " << i+1 << ": " << t << " queries/sec" << endl;
            logFile << "Dataset_RangeQuery," << name << "," << i+1 << "," << NUM_THREADS << "," << t << endl;
        }
        cout << "[Dataset " << name << " - Range Query] Average Throughput: " << (ds_sumRQ/iterations) << " queries/sec" << endl;
         */
    };

    runDatasetBenchmarks("Facebook (fb_200M_uint64)", path1);
    runDatasetBenchmarks("Books (books_800M_uint64)", path2);
    runDatasetBenchmarks("OSM (osm_cellids_800M_uint64)", path3);
    runDatasetBenchmarks("Wiki (wiki_ts_200M_uint64)", path4);

    logFile.close();
    cout << "\nAll multithreaded benchmarks completed. Results saved in benchmark_results_mt.csv" << endl;
    cout << "Total threads used: " << NUM_THREADS << endl;
    cout << "Note: Workload was distributed among threads for fair comparison" << endl;

    return 0;
}