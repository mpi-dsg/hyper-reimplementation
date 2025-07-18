# Hyper Index - High-Performance Learned Index Re-Implementation

This project is a C++ reimplementation of the **Hyper** learned index structure based on the paper "Hyper: Achieving High-Performance and Memory-Efficient Learned Index Via Hybrid Construction" (2024).

## Overview

Hyper is a learned index that combines the benefits of both bottom-up and top-down construction approaches:

- **Bottom-up construction** for leaf nodes using piecewise linear approximation (PLA)
- **Top-down construction** for inner nodes using learned models
- **Hybrid approach** that optimizes the trade-off between memory efficiency and query performance

## Architecture

### Core Components

1. **Hyper Index (`hyper_index.h/.cpp`)**: Main index interface with bulk loading, point queries, insertions, and range queries
2. **Leaf Nodes (`leaf_node.h/.cpp`)**: Store key-value pairs using linear models with overflow buffers for collisions
3. **Model Inner Nodes (`model_inner_node.h/.cpp`)**: Use learned models to predict child positions
4. **Search Inner Nodes (`search_inner_node.h/.cpp`)**: Traditional B-tree-like nodes for fallback scenarios
5. **Overflow Buffers (`overflow_buffer.h/.cpp`)**: Handle collisions in leaf nodes using sorted vectors
6. **Configuration Search (`configuration_search.h`)**: Finds optimal parameters for model-based nodes

## Building

### Prerequisites

- CMake 3.27 or higher
- C++17 compatible compiler
- OpenMP support
- SSE support

### Compilation

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Operations

```cpp
#include "include/hyper_index.h"

// Create index
Hyper index;

// Bulk load data
std::vector<std::pair<uint64_t, uint64_t>> data = {{1, 100}, {2, 200}, {3, 300}};
index.bulkLoad(data);

// Point query
auto result = index.find(2);
if (result) {
    std::cout << "Found: " << *result << std::endl;
}

// Insert new key-value pair
index.insert(4, 400);

// Range query
auto range_results = index.rangeQuery(1, 3);

// GC is triggered manually (work in progress)
EpochManager::get().forceReclamation();

```

### Benchmarking

Run the included benchmark suite:

```bash
./HyperCode
```

The benchmark tests various workloads:

- Read-only queries
- Write-only insertions
- Mixed read-write operations
- Bulk loading performance
- Range queries

## Current Implementation Status

### ✅ Implemented Features

- **Core Index Structure**: Complete implementation of hybrid construction
- **Concurrent Operations**: Thread-safe operations with fine-grained locking
- **Memory Management**: Efficient memory layout minimizing cache misses
- **Overflow Handling**: Sorted overflow buffers for collision resolution
- **Benchmarking Suite**: Comprehensive performance testing framework


## Configuration Parameters

Key tunable parameters:

- `delta` (default: 128.0): Error bound for piecewise linear approximation
- `lambda` (default: 0.1): Memory oversubscription factor
- `max_node_size` (default: 16MB): Maximum node size in bytes
- `NUM_THREADS` (default: 8): Number of threads for parallel operations
- `USE_EPOCHS` (default: false): Turn on Epoch Based GC. Currently wip.

## References

**Original Paper:**

- **Title**: "Hyper: A High-Performance and Memory-Efficient Learned Index via Hybrid Construction"
- **Authors**: S. Zhang, J. Qi, X. Yao, A. Brinkmann
- **Conference**: ACM Transactions on Management of Data (TODS), 2024
- **DOI**: [10.1145/3654948](https://dl.acm.org/doi/abs/10.1145/3654948)

## License

This implementation is provided for research and educational purposes. Please refer to the original paper for algorithmic details and cite appropriately if used in academic work.

## Acknowledgments

The largest part of this reimplementation was developed by our bachelor student Sven Weber. We thank him for his significant contributions to the project.