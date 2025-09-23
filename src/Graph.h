#pragma once

// Sara Stojkov SV38/2023

#include <vector>
#include <map>
#include <string>

/**
 * @brief Implementation of a directed graph used for PageRank calculation and representation.
 */
class Graph {
public:
    // Constructors & Destructor
    Graph();                 // empty constructor
    Graph(int n);            // construct with n nodes
    ~Graph();                // destructor

    // Basic graph operations
    void addEdge(int from, int to);
    const std::vector<int>& getInNeighbors(int v) const;
    int getOutDegree(int v) const;

    // Random graph generation
    void generateErdosRenyi(double p, unsigned int seed = 0);
    void generateErdosRenyiWithEdges(int targetEdges, unsigned int seed = 0);

    // Analysis & stats
    void printGraphStats() const;
    int getTotalEdges() const;
    double getAverageDegree() const;

    // Utility
    void clear();
    int numNodes() const { return N; }

private:
    int N;  // number of nodes
    std::vector<std::vector<int>> adjListIn;   // adjacency list (incoming edges)
    std::vector<int> outDegree;                // out-degree counts
    std::map<std::string, int> names;          // optional mapping name -> node id
};
