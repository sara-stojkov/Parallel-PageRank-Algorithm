#pragma once

// Sara Stojkov SV38/2023

#include "Graph.h"
#include <vector>

/**
 * @brief PageRank algorithm with different versions, based on the Graph class and parallelisation.
 */
class PageRank {
public:
    PageRank(const Graph& g, double d = 0.85);

    // Sequential version
    void runSequential(int maxIter, double epsilon);

    // Parallel version
    void runParallel(int maxIter, double epsilon, int numThreads);

    // Additional optimized versions for testing
    void runParallelBlockOptimized(int maxIter, double epsilon, int numThreads);
    void runParallelVectorized(int maxIter, double epsilon, int numThreads);
    void runParallelNUMAAware(int maxIter, double epsilon, int numThreads);
    void runParallelPushBased(int maxIter, double epsilon, int numThreads);
    void runParallelAdaptiveGrain(int maxIter, double epsilon, int numThreads);

    // Enhanced versions with importance modeling
    void runWithPersonalization(int maxIter, double epsilon, int numThreads,
        const std::vector<double>& personalization_vector);
    void runWithNodeWeights(int maxIter, double epsilon, int numThreads,
        const std::vector<double>& node_weights);
    void runWithEdgeWeights(int maxIter, double epsilon, int numThreads,
        const std::vector<std::vector<double>>& edge_weights);

    // Getter for results
    const std::vector<double>& getRanks() const { return prOld; }

    // Utility functions for analysis
    void printConvergenceStats() const;
    void printTopNodes(int k = 10) const;
    void analyzeImportanceDistribution() const;
    std::vector<int> getTopKNodes(int k) const;
    double calculateL1Norm() const;

private:
    const Graph& graph;
    double damping;                  // damping factor
    std::vector<double> prOld, prNew;

    // Helper functions
    bool hasConverged(double epsilon) const;
    bool hasConvergedRelative(double epsilon) const;
    void initialize();
};