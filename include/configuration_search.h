#ifndef HYPERCODE_CONFIGURATION_SEARCH_H
#define HYPERCODE_CONFIGURATION_SEARCH_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include "common_defs.h"

/**
 * @class ConfigurationSearch
 * @brief Finds optimal parameters for model-based inner nodes
 *
 * ConfigurationSearch is responsible for finding the optimal slope and model range (MR)
 * for a model-based inner node by evaluating different configurations and selecting
 * the one with the lowest cost.
 */
class ConfigurationSearch {
public:
    /**
     * @struct Result
     * @brief Contains the optimal configuration parameters found by the search
     */
    struct Result {
        double best_slope;              ///< Optimal slope for the linear model
        size_t best_mr;                 ///< Optimal model range
        double best_cost;               ///< Cost associated with the optimal configuration
        std::vector<int> best_slot_counts; ///< Slot occupancy distribution for optimal configuration
    };

    /**
     * @brief Constructs a configuration search for a set of keys
     * @param keys Vector of keys to configure the model for
     * @param lambda Memory oversubscription factor
     */
    ConfigurationSearch(const std::vector<KeyType>& keys, double lambda)
            : keys_(keys), lambda_(lambda) {}

    /**
     * @brief Executes the search to find the optimal configuration
     * @return Result object containing the optimal parameters
     */
    Result search() {
        if (keys_.empty()) return {0, 0, 0};

        const size_t N = keys_.size();
        const KeyType max_diff = keys_.back() - keys_.front();

        // Initialize with a reasonable slope
        double slope = compute_initial_slope();
        double best_slope = slope;
        double best_cost = std::numeric_limits<double>::max();
        size_t best_mr = 0;
        size_t over_sub_factor = ceil(N * (1 + lambda_));
        std::vector<int> best_slot_counts;

        // Number of search iterations based on input size
        auto search_iter = static_cast<size_t>(log2(N) + 1);

        double prev_slope = 0;
        double prev_cost = std::numeric_limits<double>::max();

        // Exponential search phase
        for (size_t i = 0; i < search_iter; ++i) {
            // Calculate model range based on current slope
            size_t mr = std::min(over_sub_factor, (size_t)ceil(best_slope * max_diff));

            // Evaluate the cost of the current configuration
            auto [cost, slot_counts] = compute_cost(slope, mr);

            // Update best configuration if current is better
            if (cost < best_cost) {
                best_cost = cost;
                best_slope = slope;
                best_mr = mr;
                best_slot_counts = slot_counts;
            }

            // Decide next step based on cost improvement
            if (cost < prev_cost) {
                // Cost is still improving, continue decreasing slope
                slope /= 2;
            }
            else {
                // Cost is no longer improving, perform binary search
                double low = prev_slope;
                double high = best_slope;
                for (size_t j = i; j < search_iter; ++j) {
                    double mid = (low + high) / 2;
                    size_t mid_mr = std::min(over_sub_factor, (size_t)ceil(mid * max_diff));
                    auto [mid_cost, mid_slot_counts] = compute_cost(mid, mid_mr);

                    if (mid_cost < best_cost) {
                        best_cost = mid_cost;
                        best_slope = mid;
                        best_mr = mid_mr;
                        best_slot_counts = mid_slot_counts;
                        high = mid;
                    } else {
                        low = mid;
                    }
                }
                break;
            }
            prev_cost = cost;
            prev_slope = slope;
        }
        return {best_slope, best_mr, best_cost, best_slot_counts};
    }

private:
    /**
     * @brief Computes an initial slope based on median key differences
     * @return Initial slope value
     */
    double compute_initial_slope() {
        const size_t n = keys_.size();
        if (n < 2) return 1.0;

        // Calculate differences between consecutive keys
        std::vector<KeyType> diffs;
        diffs.reserve(n - 1);

        for (size_t i = 1; i < n; ++i) {
            diffs.push_back(keys_[i] - keys_[i-1]);
        }

        // Find the median difference
        std::nth_element(diffs.begin(), diffs.begin() + diffs.size() / 2, diffs.end());
        auto median_diff = static_cast<double>(diffs[diffs.size()/2]);

        // Return reciprocal of median difference as initial slope
        return median_diff > 0 ? 1.0 / median_diff : 0;
    }

    /**
     * @brief Computes the cost of a configuration with given slope and model range
     * @param slope Slope to evaluate
     * @param mr Model range to evaluate
     * @return Pair of cost value and slot counts
     */
    std::pair<double, std::vector<int>> compute_cost(double slope, size_t mr) {
        // Initialize slot count array and assign keys to slots
        std::vector<int> slot_counts(mr + 1, 0);
        KeyType base = keys_.front();

        for (KeyType key : keys_) {
            size_t pos = ceil(slope * static_cast<double>(key - base));
            pos = std::min(pos, mr);
            slot_counts[pos]++;
        }

        // Calculate statistics for cost computation
        size_t empty_slots = 0;
        size_t accurate_slots = 0;
        size_t search_slots = 0;
        size_t model_slots = 0;
        size_t conflict_sum = 0;
        std::vector<int> conflicts;

        for (int count : slot_counts) {
            if (count == 0) {
                empty_slots++;
            } else if (count == 1) {
                accurate_slots++;
            } else if (count <= 8) {
                search_slots++;
                conflict_sum += count;
                conflicts.push_back(count);
            } else {
                model_slots++;
                conflict_sum += count;
                conflicts.push_back(count);
            }
        }

        const size_t total_slots = slot_counts.size();
        const size_t non_empty = total_slots - empty_slots;
        const size_t search_model_slots = total_slots - (empty_slots + accurate_slots);

        // Standard deviation of conflicts
        double mean = (search_slots + model_slots == 0) ? 0 :
                      conflict_sum / static_cast<double>(search_slots + model_slots);
        double variance = 0;
        for (auto c : conflicts) {
            variance += (c - mean) * (c - mean);
        }
        double std_dev = (conflicts.size() == 0) ? 0 :
                         sqrt(variance / conflicts.size());

        // Normalized conflict cost
        double conflict_cost = (search_model_slots == 0) ? 0 :
                               conflict_sum / static_cast<double>(search_model_slots);
        double norm_conflict = conflict_cost / non_empty;

        // Empty slots ratio
        double empty_ratio = static_cast<double>(empty_slots) / total_slots;

        // Final cost combining all factors
        double cost = std_dev + norm_conflict + empty_ratio;

        return {cost, slot_counts};
    }

private:
    const std::vector<KeyType>& keys_; ///< Reference to keys to configure model for
    const double lambda_;              ///< Memory oversubscription factor
};

#endif //HYPERCODE_CONFIGURATION_SEARCH_H