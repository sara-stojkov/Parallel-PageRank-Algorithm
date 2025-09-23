#pragma once

// Sara Stojkov SV38/2023

// Sara Stojkov SV38/2023
#pragma once

#include <vector>
#include <string>
#include "Graph.h"
#include "PageRank.h"

/**
 * @brief Stores the results of a single benchmark test run.
 */
struct BenchmarkResult {
    std::string test_name;
    int nodes;
    int edges;
    double avg_degree;
    double sequential_time_ms;
    double parallel_time_ms;
    double speedup;
    int threads;
    bool converged_sequential;
    bool converged_parallel;
    double max_rank_diff;
};

/**
 * @brief Benchmark suite for running and comparing different PageRank implementations.
 */
class PageRankBenchmark {
private:
    std::vector<BenchmarkResult> results; ///< Stores all benchmark results

public:
    void runBenchmarkSuite();
    void runSingleBenchmark(int nodes, double edge_probability);
    void testOptimizedVersions(int nodes = 10000, double p = 0.005);
    void printSummaryReport() const;
    void exportResultsToCSV(const std::string& filename) const;
};
