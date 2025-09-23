// Sara Stojkov SV38/2023

#include "PageRankBenchmark.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <chrono>
#include <functional>
#include <algorithm>

/**
 * @brief Runs the full benchmark suite with predefined graph sizes and densities. Handles all implementations and saves results. 
 */

void PageRankBenchmark::runBenchmarkSuite() {
    std::cout << "PAGERANK PERFORMANCE BENCHMARK SUITE\n";
    std::cout << "========================================\n\n";

    std::vector<int> node_counts = { 1000, 5000, 10000, 50000, 100000 };
    std::vector<double> densities = { 0.001, 0.005, 0.01, 0.02 };

    int total_tests = node_counts.size() * densities.size();
    int current_test = 0;

    for (int nodes : node_counts) {
        for (double p : densities) {
            current_test++;
            std::cout << " TEST " << current_test << "/" << total_tests
                << ": " << nodes << " nodes, p=" << p << "\n";
            std::cout << "Expected edges: ~" << static_cast<int>(p * nodes * (nodes - 1)) << "\n";

            runSingleBenchmark(nodes, p);
            std::cout << " Test completed\n";
            std::cout << std::string(50, '-') << "\n\n";
        }
    }

    printSummaryReport();
    exportResultsToCSV("pagerank_benchmark_results.csv");
}

/**
 * @brief Runs a single benchmark test on a generated Erd?s–Rényi random graph.
 * @param nodes Number of nodes in the graph.
 * @param edge_probability Probability of edge creation between nodes.
 */
void PageRankBenchmark::runSingleBenchmark(int nodes, double edge_probability) {
    Graph graph(nodes);
    graph.generateErdosRenyi(edge_probability, 12345);
    graph.printGraphStats();

    const int max_iterations = 100;
    const double epsilon = 1e-6;
    const int num_threads = std::thread::hardware_concurrency();

    BenchmarkResult result;
    result.test_name = "ErdosRenyi_" + std::to_string(nodes) + "_p" + std::to_string(edge_probability);
    result.nodes = nodes;
    result.edges = graph.getTotalEdges();
    result.avg_degree = graph.getAverageDegree();
    result.threads = num_threads;

    // Sequential PageRank
    std::cout << " Running SEQUENTIAL PageRank...\n";
    PageRank pr_sequential(graph, 0.85);
    auto start = std::chrono::high_resolution_clock::now();
    pr_sequential.runSequential(max_iterations, epsilon);
    auto end = std::chrono::high_resolution_clock::now();

    result.sequential_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.converged_sequential = true;

    std::cout << "   Sequential completed in " << std::fixed << std::setprecision(2)
        << result.sequential_time_ms << " ms\n";

    // Parallel PageRank
    std::cout << " Running PARALLEL PageRank (" << num_threads << " threads)...\n";
    PageRank pr_parallel(graph, 0.85);
    start = std::chrono::high_resolution_clock::now();
    pr_parallel.runParallel(max_iterations, epsilon, num_threads);
    end = std::chrono::high_resolution_clock::now();

    result.parallel_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.converged_parallel = true;
    result.speedup = result.sequential_time_ms / result.parallel_time_ms;

    std::cout << "   Parallel completed in " << std::fixed << std::setprecision(2)
        << result.parallel_time_ms << " ms\n";
    std::cout << "    SPEEDUP: " << std::fixed << std::setprecision(2)
        << result.speedup << "x\n";

    // Verify correctness
    const auto& seq_ranks = pr_sequential.getRanks();
    const auto& par_ranks = pr_parallel.getRanks();

    double max_diff = 0.0;
    for (size_t i = 0; i < seq_ranks.size(); ++i)
        max_diff = std::max(max_diff, std::abs(seq_ranks[i] - par_ranks[i]));

    result.max_rank_diff = max_diff;
    std::cout << "   Max difference between sequential/parallel: "
        << std::scientific << max_diff << "\n";

    if (max_diff < 1e-10)
        std::cout << "   Results are numerically identical!\n";
    else if (max_diff < 1e-6)
        std::cout << "   Results are very close (within tolerance)\n";
    else
        std::cout << "   Results differ more than expected\n";

    if (nodes <= 1000) {
        std::cout << "\nTOP 5 MOST IMPORTANT NODES:\n";
        pr_parallel.printTopNodes(5);
    }

    results.push_back(result);
}

/**
 * @brief Runs and compares different PageRank optimization strategies.
 * @param nodes Number of nodes in the test graph (default: 10000).
 * @param p Edge probability for the Erd?s–Rényi random graph (default: 0.005).
 */
void PageRankBenchmark::testOptimizedVersions(int nodes, double p) {
    std::cout << "\n TESTING ALL OPTIMIZATION VARIANTS\n";
    std::cout << "====================================\n";

    Graph graph(nodes);
    graph.generateErdosRenyi(p, 42);
    graph.printGraphStats();

    const int max_iter = 50;
    const double epsilon = 1e-6;
    const int threads = std::thread::hardware_concurrency();

    struct OptTest {
        std::string name;
        std::function<void(PageRank&)> function;
        double time_ms;
    };

    std::vector<OptTest> tests = {
        {"Sequential", [&](PageRank& pr) { pr.runSequential(max_iter, epsilon); }},
        {"Parallel Basic", [&](PageRank& pr) { pr.runParallel(max_iter, epsilon, threads); }},
        {"Block Optimized", [&](PageRank& pr) { pr.runParallelBlockOptimized(max_iter, epsilon, threads); }},
        {"Vectorized", [&](PageRank& pr) { pr.runParallelVectorized(max_iter, epsilon, threads); }},
        {"NUMA Aware", [&](PageRank& pr) { pr.runParallelNUMAAware(max_iter, epsilon, threads); }},
        {"Adaptive Grain", [&](PageRank& pr) { pr.runParallelAdaptiveGrain(max_iter, epsilon, threads); }}
    };

    std::vector<double> baseline_ranks;

    for (auto& test : tests) {
        std::cout << "Testing: " << test.name << "...\n";
        PageRank pr(graph, 0.85);

        auto start = std::chrono::high_resolution_clock::now();
        test.function(pr);
        auto end = std::chrono::high_resolution_clock::now();

        test.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "   Time: " << std::fixed << std::setprecision(2) << test.time_ms << " ms\n";

        if (test.name == "Sequential")
            baseline_ranks = pr.getRanks();
        else {
            const auto& current_ranks = pr.getRanks();
            double max_diff = 0.0;
            for (size_t i = 0; i < baseline_ranks.size(); ++i)
                max_diff = std::max(max_diff, std::abs(baseline_ranks[i] - current_ranks[i]));

            std::cout << "    Max diff from sequential: " << std::scientific << max_diff << "\n";
            std::cout << "    Speedup: " << std::fixed << std::setprecision(2)
                << (tests[0].time_ms / test.time_ms) << "x\n";
        }
        std::cout << "\n";
    }

    std::cout << "OPTIMIZATION COMPARISON SUMMARY:\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << std::left << std::setw(18) << "Method"
        << std::setw(12) << "Time (ms)"
        << std::setw(12) << "Speedup"
        << "Efficiency\n";
    std::cout << std::string(60, '-') << "\n";

    for (size_t i = 0; i < tests.size(); ++i) {
        double speedup = tests[0].time_ms / tests[i].time_ms;
        double efficiency = speedup / threads * 100.0;

        std::cout << std::left << std::setw(18) << tests[i].name
            << std::setw(12) << std::fixed << std::setprecision(2) << tests[i].time_ms
            << std::setw(12) << std::fixed << std::setprecision(2) << speedup;

        if (i == 0)
            std::cout << "baseline";
        else
            std::cout << std::fixed << std::setprecision(1) << efficiency << "%";

        std::cout << "\n";
    }
    std::cout << "\n";
}

/**
 * @brief Prints a summary report of all collected benchmark results to the console.
 */
void PageRankBenchmark::printSummaryReport() const {
    std::cout << "\n BENCHMARK SUMMARY REPORT\n";
    std::cout << "===========================\n\n";

    std::cout << std::left << std::setw(8) << "Nodes"
        << std::setw(8) << "Edges"
        << std::setw(8) << "Seq(ms)"
        << std::setw(8) << "Par(ms)"
        << std::setw(10) << "Speedup"
        << std::setw(12) << "Efficiency"
        << "Max Diff\n";
    std::cout << std::string(70, '-') << "\n";

    double total_speedup = 0.0;
    int valid_tests = 0;

    for (const auto& r : results) {
        if (r.speedup > 0) {
            total_speedup += r.speedup;
            valid_tests++;
        }

        double efficiency = (r.speedup / r.threads) * 100.0;

        std::cout << std::left << std::setw(8) << r.nodes
            << std::setw(8) << r.edges
            << std::setw(8) << std::fixed << std::setprecision(1) << r.sequential_time_ms
            << std::setw(8) << std::fixed << std::setprecision(1) << r.parallel_time_ms
            << std::setw(10) << std::fixed << std::setprecision(2) << r.speedup
            << std::setw(12) << std::fixed << std::setprecision(1) << efficiency << "%"
            << std::scientific << r.max_rank_diff << "\n";
    }

    if (valid_tests > 0) {
        std::cout << std::string(70, '-') << "\n";
        std::cout << "Average speedup: " << std::fixed << std::setprecision(2)
            << (total_speedup / valid_tests) << "x\n";
        std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    }
}

/**
 * @brief Exports benchmark results to a CSV file.
 *  @param filename Name of the output CSV file.
 */
void PageRankBenchmark::exportResultsToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing\n";
        return;
    }

    file << "test_name,nodes,edges,avg_degree,sequential_ms,parallel_ms,speedup,threads,max_diff\n";

    for (const auto& r : results) {
        file << r.test_name << ","
            << r.nodes << ","
            << r.edges << ","
            << std::fixed << std::setprecision(3) << r.avg_degree << ","
            << std::fixed << std::setprecision(2) << r.sequential_time_ms << ","
            << std::fixed << std::setprecision(2) << r.parallel_time_ms << ","
            << std::fixed << std::setprecision(3) << r.speedup << ","
            << r.threads << ","
            << std::scientific << r.max_rank_diff << "\n";
    }

    file.close();
    std::cout << "  Results exported to " << filename << "\n";
}
