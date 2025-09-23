// Sara Stojkov SV38/2023

#include "PageRank.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

/**
 * @brief Construct a PageRank object for the given graph and damping factor.
 * @param g The input graph.
 * @param d Damping factor for PageRank calculation.
 */
PageRank::PageRank(const Graph& g, double d)
    : graph(g), damping(d) {
    initialize();
}

/**
 * @brief Initialize PageRank vectors with default values.
 */
void PageRank::initialize() {
    int n = graph.numNodes();
    prOld.resize(n, 1.0 / n);  // Initialize with uniform distribution
    prNew.resize(n, 0.0);
}

/**
 * @brief Run sequential PageRank algorithm.
 * @param maxIter Maximum number of iterations.
 * @param epsilon Convergence parameter.
 */
void PageRank::runSequential(int maxIter, double epsilon) {
    int n = graph.numNodes();
    double base_rank = (1.0 - damping) / n;

    for (int iter = 0; iter < maxIter; ++iter) {
        // Reset new PageRank values
        std::fill(prNew.begin(), prNew.end(), base_rank);

        // Compute new PageRank values
        for (int v = 0; v < n; ++v) {
            const std::vector<int>& in_neighbors = graph.getInNeighbors(v);
            double rank_sum = 0.0;

            for (int u : in_neighbors) {
                int out_deg = graph.getOutDegree(u);
                if (out_deg > 0) {
                    rank_sum += prOld[u] / out_deg;
                }
            }

            prNew[v] += damping * rank_sum;
        }

        // Check for convergence
        if (hasConverged(epsilon)) {
            break;
        }

        // Swap arrays for next iteration
        prOld.swap(prNew);
    }

    // Ensure prOld contains the final results
    if (prOld.empty() || prOld[0] == 0.0) {
        prOld.swap(prNew);
    }
}

/**
 * @brief Run parallel PageRank algorithm using TBB.
 * @param maxIter Maximum number of iterations.
 * @param epsilon Convergence threshold.
 * @param numThreads Number of threads to use.
 */
void PageRank::runParallel(int maxIter, double epsilon, int numThreads) {
    // Modern TBB thread control
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, numThreads);

    int n = graph.numNodes();
    double base_rank = (1.0 - damping) / n;

    // Calculate optimal grain size
    const int grain_size = std::max(1, n / (4 * numThreads));

    for (int iter = 0; iter < maxIter; ++iter) {
        // Reset new PageRank values in parallel
        tbb::parallel_for(tbb::blocked_range<int>(0, n, grain_size),
            [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i != range.end(); ++i) {
                    prNew[i] = base_rank;
                }
            });

        // Compute new PageRank values in parallel
        tbb::parallel_for(tbb::blocked_range<int>(0, n, grain_size),
            [&](const tbb::blocked_range<int>& range) {
                for (int v = range.begin(); v != range.end(); ++v) {
                    const std::vector<int>& in_neighbors = graph.getInNeighbors(v);
                    double rank_sum = 0.0;

                    for (int u : in_neighbors) {
                        int out_deg = graph.getOutDegree(u);
                        if (out_deg > 0) {
                            rank_sum += prOld[u] / out_deg;
                        }
                    }

                    prNew[v] += damping * rank_sum;
                }
            });

        // Check for convergence in parallel
        if (hasConverged(epsilon)) {
            break;
        }

        // Swap arrays for next iteration
        prOld.swap(prNew);
    }

    // Ensure prOld contains the final results
    if (prOld.empty() || prOld[0] == 0.0) {
        prOld.swap(prNew);
    }
}

/**
 * @brief Check convergence based on absolute difference.
 * @param epsilon Convergence threshold.
 * @return True if maximum difference is below epsilon, false otherwise.
 */
bool PageRank::hasConverged(double epsilon) const {
    int n = static_cast<int>(prOld.size());

    // Use parallel reduction to find maximum difference
    double max_diff = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, n), 0.0,
        [&](const tbb::blocked_range<int>& range, double local_max) -> double {
            double local_max_diff = 0.0;
            for (int i = range.begin(); i != range.end(); ++i) {
                double diff = std::abs(prNew[i] - prOld[i]);
                local_max_diff = std::max(local_max_diff, diff);
            }
            return std::max(local_max, local_max_diff);
        },
        [](double a, double b) -> double {
            return std::max(a, b);
        }
    );

    return max_diff < epsilon;
}

/**
 * @brief Check convergence based on relative difference.
 * @param epsilon Convergence threshold.
 * @return True if maximum relative difference is below epsilon, false otherwise.
 */
bool PageRank::hasConvergedRelative(double epsilon) const {
    int n = static_cast<int>(prOld.size());

    double max_relative_diff = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, n), 0.0,
        [&](const tbb::blocked_range<int>& range, double local_max) -> double {
            double local_max_diff = 0.0;
            for (int i = range.begin(); i != range.end(); ++i) {
                if (prOld[i] > 0.0) {
                    double relative_diff = std::abs(prNew[i] - prOld[i]) / prOld[i];
                    local_max_diff = std::max(local_max_diff, relative_diff);
                }
            }
            return std::max(local_max, local_max_diff);
        },
        [](double a, double b) -> double {
            return std::max(a, b);
        }
    );

    return max_relative_diff < epsilon;
}

/**
 * @brief Run parallel block-optimized PageRank algorithm.
 * @param maxIter Maximum number of iterations.
 * @param epsilon Convergence threshold.
 * @param numThreads Number of threads to use.
 */
void PageRank::runParallelBlockOptimized(int maxIter, double epsilon, int numThreads) {
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, numThreads);

    int n = graph.numNodes();
    double base_rank = (1.0 - damping) / n;

    const int CACHE_LINE_SIZE = 64;
    const int doubles_per_cache_line = CACHE_LINE_SIZE / sizeof(double);
    const int block_size = std::max(doubles_per_cache_line, n / (numThreads * 4));

    for (int iter = 0; iter < maxIter; ++iter) {
        tbb::parallel_for(tbb::blocked_range<int>(0, n, block_size),
            [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i < range.end(); i += doubles_per_cache_line) {
                    int end_idx = std::min(i + doubles_per_cache_line, range.end());
                    for (int j = i; j < end_idx; ++j) {
                        prNew[j] = base_rank;
                    }
                }
            });

        tbb::parallel_for(tbb::blocked_range<int>(0, n, block_size),
            [&](const tbb::blocked_range<int>& range) {
                for (int v = range.begin(); v != range.end(); ++v) {
                    const std::vector<int>& in_neighbors = graph.getInNeighbors(v);
                    double rank_sum = 0.0;

                    size_t neighbor_count = in_neighbors.size();
                    size_t unroll_limit = neighbor_count - (neighbor_count % 4);

                    size_t i = 0;
                    for (; i < unroll_limit; i += 4) {
                        double sum_chunk = 0.0;
                        for (int k = 0; k < 4; ++k) {
                            int u = in_neighbors[i + k];
                            int out_deg = graph.getOutDegree(u);
                            if (out_deg > 0) {
                                sum_chunk += prOld[u] / out_deg;
                            }
                        }
                        rank_sum += sum_chunk;
                    }

                    for (; i < neighbor_count; ++i) {
                        int u = in_neighbors[i];
                        int out_deg = graph.getOutDegree(u);
                        if (out_deg > 0) {
                            rank_sum += prOld[u] / out_deg;
                        }
                    }

                    prNew[v] += damping * rank_sum;
                }
            });

        if (hasConverged(epsilon)) break;
        prOld.swap(prNew);
    }
}

/**
 * @brief Run parallel vectorized PageRank algorithm.
 * @param maxIter Maximum number of iterations.
 * @param epsilon Convergence threshold.
 * @param numThreads Number of threads to use.
 */
void PageRank::runParallelVectorized(int maxIter, double epsilon, int numThreads) {
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, numThreads);

    int n = graph.numNodes();
    double base_rank = (1.0 - damping) / n;

    for (int iter = 0; iter < maxIter; ++iter) {
        tbb::parallel_for(tbb::blocked_range<int>(0, n),
            [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i != range.end(); ++i) {
                    prNew[i] = base_rank;
                }
            });

        tbb::parallel_for(tbb::blocked_range<int>(0, n),
            [&](const tbb::blocked_range<int>& range) {
                for (int v = range.begin(); v != range.end(); ++v) {
                    const std::vector<int>& in_neighbors = graph.getInNeighbors(v);
                    double rank_sum = 0.0;

                    size_t neighbor_count = in_neighbors.size();
                    size_t vec_limit = neighbor_count - (neighbor_count % 4);

                    for (size_t i = 0; i < vec_limit; i += 4) {
                        double contributions[4];
                        for (int k = 0; k < 4; ++k) {
                            int u = in_neighbors[i + k];
                            int out_deg = graph.getOutDegree(u);
                            contributions[k] = (out_deg > 0) ? (prOld[u] / out_deg) : 0.0;
                        }
                        rank_sum += contributions[0] + contributions[1] + contributions[2] + contributions[3];
                    }

                    for (size_t i = vec_limit; i < neighbor_count; ++i) {
                        int u = in_neighbors[i];
                        int out_deg = graph.getOutDegree(u);
                        if (out_deg > 0) {
                            rank_sum += prOld[u] / out_deg;
                        }
                    }

                    prNew[v] += damping * rank_sum;
                }
            });

        if (hasConverged(epsilon)) break;
        prOld.swap(prNew);
    }
}

/**
 * @brief Run parallel NUMA-aware PageRank algorithm.
 * @param maxIter Maximum number of iterations.
 * @param epsilon Convergence threshold.
 * @param numThreads Number of threads to use.
 */
void PageRank::runParallelNUMAAware(int maxIter, double epsilon, int numThreads) {
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, numThreads);

    int n = graph.numNodes();
    double base_rank = (1.0 - damping) / n;

    const int nodes_per_core = (n + numThreads - 1) / numThreads;

    for (int iter = 0; iter < maxIter; ++iter) {
        tbb::parallel_for(tbb::blocked_range<int>(0, numThreads),
            [&](const tbb::blocked_range<int>& core_range) {
                for (int core_id = core_range.begin(); core_id < core_range.end(); ++core_id) {
                    int start_node = core_id * nodes_per_core;
                    int end_node = std::min((core_id + 1) * nodes_per_core, n);

                    for (int i = start_node; i < end_node; ++i) {
                        prNew[i] = base_rank;
                    }
                }
            });

        tbb::parallel_for(tbb::blocked_range<int>(0, numThreads),
            [&](const tbb::blocked_range<int>& core_range) {
                for (int core_id = core_range.begin(); core_id < core_range.end(); ++core_id) {
                    int start_node = core_id * nodes_per_core;
                    int end_node = std::min((core_id + 1) * nodes_per_core, n);

                    for (int v = start_node; v < end_node; ++v) {
                        const std::vector<int>& in_neighbors = graph.getInNeighbors(v);
                        double rank_sum = 0.0;

                        for (int u : in_neighbors) {
                            int out_deg = graph.getOutDegree(u);
                            if (out_deg > 0) {
                                rank_sum += prOld[u] / out_deg;
                            }
                        }

                        prNew[v] += damping * rank_sum;
                    }
                }
            });

        if (hasConverged(epsilon)) break;
        prOld.swap(prNew);
    }
}

/**
 * @brief Run parallel PageRank with adaptive grain size.
 * @param maxIter Maximum number of iterations.
 * @param epsilon Convergence threshold.
 * @param numThreads Number of threads to use.
 */
void PageRank::runParallelAdaptiveGrain(int maxIter, double epsilon, int numThreads) {
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, numThreads);

    int n = graph.numNodes();
    double base_rank = (1.0 - damping) / n;

    std::vector<int> work_estimates(n);
    for (int i = 0; i < n; ++i) {
        work_estimates[i] = static_cast<int>(graph.getInNeighbors(i).size()) + 1;
    }

    std::vector<long long> cumulative_work(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        cumulative_work[i + 1] = cumulative_work[i] + work_estimates[i];
    }

    long long total_work = cumulative_work[n];
    long long work_per_thread = total_work / numThreads;

    for (int iter = 0; iter < maxIter; ++iter) {
        tbb::parallel_for(tbb::blocked_range<int>(0, numThreads),
            [&](const tbb::blocked_range<int>& thread_range) {
                for (int tid = thread_range.begin(); tid < thread_range.end(); ++tid) {
                    long long target_work_start = tid * work_per_thread;
                    long long target_work_end = (tid + 1) * work_per_thread;

                    int start_node = static_cast<int>(std::lower_bound(cumulative_work.begin(), cumulative_work.end(),
                        target_work_start) - cumulative_work.begin());
                    int end_node = static_cast<int>(std::lower_bound(cumulative_work.begin(), cumulative_work.end(),
                        target_work_end) - cumulative_work.begin());

                    start_node = std::max(0, std::min(start_node, n));
                    end_node = std::max(0, std::min(end_node, n));

                    for (int i = start_node; i < end_node; ++i) {
                        prNew[i] = base_rank;

                        const std::vector<int>& in_neighbors = graph.getInNeighbors(i);
                        double rank_sum = 0.0;

                        for (int u : in_neighbors) {
                            int out_deg = graph.getOutDegree(u);
                            if (out_deg > 0) {
                                rank_sum += prOld[u] / out_deg;
                            }
                        }

                        prNew[i] += damping * rank_sum;
                    }
                }
            });

        if (hasConverged(epsilon)) break;
        prOld.swap(prNew);
    }
}

/**
 * @brief Print basic convergence statistics of the PageRank vectors.
 */
void PageRank::printConvergenceStats() const {
    int n = static_cast<int>(prOld.size());
    double sum = 0.0, max_val = 0.0, min_val = 1.0;

    for (int i = 0; i < n; ++i) {
        sum += prOld[i];
        max_val = std::max(max_val, prOld[i]);
        min_val = std::min(min_val, prOld[i]);
    }

    std::cout << "PageRank Statistics:\n";
    std::cout << "Sum: " << sum << " (should be ~1.0)\n";
    std::cout << "Max: " << max_val << "\n";
    std::cout << "Min: " << min_val << "\n";
    std::cout << "Avg: " << sum / n << "\n";
}


/**
 * @brief Print top-k nodes with highest PageRank scores.
 * @param k Number of top nodes to display.
 */
void PageRank::printTopNodes(int k) const {
    std::vector<std::pair<double, int>> rank_pairs;
    for (int i = 0; i < static_cast<int>(prOld.size()); ++i) {
        rank_pairs.emplace_back(prOld[i], i);
    }

    std::sort(rank_pairs.rbegin(), rank_pairs.rend());

    std::cout << "\nTOP " << k << " MOST IMPORTANT NODES:\n";
    std::cout << "Rank\tNode ID\tPageRank Score\n";
    std::cout << "----\t-------\t--------------\n";

    for (int i = 0; i < std::min(k, static_cast<int>(rank_pairs.size())); ++i) {
        double score = rank_pairs[i].first;
        int node_id = rank_pairs[i].second;

        std::cout << (i + 1) << "\t" << node_id << "\t"
            << std::fixed << std::setprecision(6) << score << "\n";
    }
    std::cout << "\n";
}

/**
 * @brief Return indices of top-k nodes based on PageRank scores.
 * @param k Number of top nodes to return.
 * @return Vector of node indices corresponding to top-k nodes.
 */
std::vector<int> PageRank::getTopKNodes(int k) const {
    std::vector<std::pair<double, int>> rank_pairs;
    for (int i = 0; i < static_cast<int>(prOld.size()); ++i) {
        rank_pairs.emplace_back(prOld[i], i);
    }

    std::sort(rank_pairs.rbegin(), rank_pairs.rend());

    std::vector<int> top_nodes;
    for (int i = 0; i < std::min(k, static_cast<int>(rank_pairs.size())); ++i) {
        top_nodes.push_back(rank_pairs[i].second);
    }

    return top_nodes;
}


/**
 * @brief Analyze and print the importance distribution of all nodes.
 */
void PageRank::analyzeImportanceDistribution() const {
    int n = static_cast<int>(prOld.size());
    std::vector<double> sorted_ranks = prOld;
    std::sort(sorted_ranks.rbegin(), sorted_ranks.rend());

    std::cout << "\nIMPORTANCE DISTRIBUTION ANALYSIS:\n";
    std::cout << "=================================\n";

    double sum = 0.0, sum_sq = 0.0;
    for (double rank : sorted_ranks) {
        sum += rank;
        sum_sq += rank * rank;
    }

    double mean = sum / n;
    double variance = (sum_sq / n) - (mean * mean);
    double std_dev = std::sqrt(variance);

    std::cout << "Basic Statistics:\n";
    std::cout << "   Mean importance: " << std::scientific << mean << "\n";
    std::cout << "   Std deviation:   " << std_dev << "\n";
    std::cout << "   Max importance:  " << sorted_ranks[0] << "\n";
    std::cout << "   Min importance:  " << sorted_ranks[n - 1] << "\n";

    if (sorted_ranks[n - 1] > 0) {
        std::cout << "   Range ratio:     " << (sorted_ranks[0] / sorted_ranks[n - 1]) << ":1\n";
    }
}

/**
 * @brief Calculate the L1 norm of the PageRank vector.
 * @return L1 norm of prOld vector.
 */
double PageRank::calculateL1Norm() const {
    int n = static_cast<int>(prOld.size());
    double norm = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, n), 0.0,
        [&](const tbb::blocked_range<int>& range, double local_sum) -> double {
            for (int i = range.begin(); i < range.end(); ++i) {
                local_sum += std::abs(prOld[i]);
            }
            return local_sum;
        },
        [](double a, double b) -> double {
            return a + b;
        }
    );
    return norm;
}