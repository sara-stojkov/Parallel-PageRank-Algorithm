// Sara Stojkov SV38/2023

#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <iomanip>
#include <fstream>
#include <string>
#include <functional>
#include "Graph.h"
#include "PageRank.h"
#include "PageRankBenchmark.h"

int main() {
    std::cout << " PARALLEL PAGERANK BENCHMARKING SUITE\n";
    std::cout << "Hardware: " << std::thread::hardware_concurrency() << " threads available\n\n";

    PageRankBenchmark benchmark;

    try {
        // Run comprehensive benchmark suite
        benchmark.runBenchmarkSuite();

        // Test all optimization variants on a medium-sized graph
        benchmark.testOptimizedVersions(5000, 0.002);

        std::cout << " All benchmarks completed successfully!\n";

    }
    catch (const std::exception& e) {
        std::cerr << " Error during benchmarking: " << e.what() << "\n";
        return 1;
    }

    return 0;
}