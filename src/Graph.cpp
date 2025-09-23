// Sara Stojkov SV38/2023

#include "Graph.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>

/// @brief Default constructor. 
/// Initializes an empty graph with zero nodes.
Graph::Graph()
    : N(0), adjListIn(), outDegree(), names() {
}

/// @brief Constructs a graph with a given number of nodes.
/// @param n Number of nodes in the graph.
Graph::Graph(int n)
    : N(n), adjListIn(n), outDegree(n, 0), names() {
}

/// @brief Destructor for the Graph class.
Graph::~Graph() {}

/// @brief Adds a directed edge from one node to another.
/// @param from Source node index.
/// @param to Destination node index.
/// @note Does nothing if indices are invalid or if edge already exists.
void Graph::addEdge(int from, int to) {
    if (from < 0 || from >= N || to < 0 || to >= N) return;

    const std::vector<int>& inNeighbors = adjListIn[to];
    if (std::find(inNeighbors.begin(), inNeighbors.end(), from) != inNeighbors.end()) {
        return; // Edge already exists
    }

    adjListIn[to].push_back(from);
    outDegree[from]++;
}

/// @brief Gets the incoming neighbors of a given node.
/// @param v Node index.
/// @return const reference to a vector of node indices pointing to v.
const std::vector<int>& Graph::getInNeighbors(int v) const {
    return adjListIn[v];
}

/// @brief Gets the out-degree (number of outgoing edges) of a node.
/// @param v Node index.
/// @return Out-degree of the given node.
int Graph::getOutDegree(int v) const {
    return outDegree[v];
}

/// @brief Generates a random graph using the Erd?s–Rényi model G(n, p).
/// @param p Probability of creating an edge between two nodes (0 ? p ? 1).
/// @param seed Random seed (0 means system-generated seed).
/// @note Clears the current graph before generation.
void Graph::generateErdosRenyi(double p, unsigned int seed) {
    if (p < 0.0 || p > 1.0) {
        std::cerr << "Error: Probability p must be between 0 and 1\n";
        return;
    }

    clear();

    std::mt19937 gen;
    if (seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    }
    else {
        gen.seed(seed);
    }

    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::cout << " Generating Erd?s-Rényi random graph...\n";
    std::cout << "   Nodes: " << N << "\n";
    std::cout << "   Edge probability: " << p << "\n";

    int edges_added = 0;

    for (int from = 0; from < N; ++from) {
        for (int to = 0; to < N; ++to) {
            if (from != to && dis(gen) < p) {
                addEdge(from, to);
                edges_added++;
            }
        }
        if (N > 10000 && from % (N / 10) == 0) {
            std::cout << "   Progress: " << (100 * from / N) << "%\n";
        }
    }

    std::cout << " Generated graph with " << edges_added << " edges\n";
    std::cout << "   Expected edges: " << static_cast<int>(p * N * (N - 1)) << "\n";
    std::cout << "   Average degree: " << getAverageDegree() << "\n\n";
}

/// @brief Generates a random graph with approximately the target number of edges.
/// @param targetEdges Desired number of edges in the graph.
/// @param seed Random seed (0 means system-generated seed).
/// @note Uses random sampling and ensures no duplicate edges.
void Graph::generateErdosRenyiWithEdges(int targetEdges, unsigned int seed) {
    if (targetEdges <= 0 || targetEdges > N * (N - 1)) {
        std::cerr << "Error: Target edges must be between 1 and " << (N * (N - 1)) << "\n";
        return;
    }

    clear();

    std::mt19937 gen;
    if (seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    }
    else {
        gen.seed(seed);
    }

    std::cout << " Generating Erd?s-Rényi graph with target edge count...\n";
    std::cout << "   Nodes: " << N << "\n";
    std::cout << "   Target edges: " << targetEdges << "\n";

    double p = static_cast<double>(targetEdges) / (N * (N - 1));
    std::cout << "   Calculated probability: " << p << "\n";

    std::unordered_set<long long> edge_set;
    std::uniform_int_distribution<int> node_dis(0, N - 1);

    int attempts = 0;
    const int max_attempts = targetEdges * 3;

    while (static_cast<int>(edge_set.size()) < targetEdges && attempts < max_attempts) {
        int from = node_dis(gen);
        int to = node_dis(gen);

        if (from != to) {
            long long edge_id = (static_cast<long long>(from) << 32) | to;
            if (edge_set.find(edge_id) == edge_set.end()) {
                edge_set.insert(edge_id);
                addEdge(from, to);
            }
        }

        attempts++;

        if (targetEdges > 10000 && attempts % (targetEdges / 10) == 0) {
            std::cout << "   Progress: " << edge_set.size() << "/" << targetEdges << " edges\n";
        }
    }

    std::cout << " Generated graph with " << getTotalEdges() << " edges\n";
    std::cout << "   Success rate: " << (100.0 * getTotalEdges() / targetEdges) << "%\n";
    std::cout << "   Average degree: " << getAverageDegree() << "\n\n";
}

/// @brief Prints statistics about the graph (nodes, edges, degree distribution, density).
/// @note Displays degree distribution only for small graphs (N ? 20).
void Graph::printGraphStats() const {
    std::cout << " GRAPH STATISTICS:\n";
    std::cout << "==================\n";
    std::cout << "Nodes: " << N << "\n";
    std::cout << "Edges: " << getTotalEdges() << "\n";
    std::cout << "Average degree: " << std::fixed << std::setprecision(2) << getAverageDegree() << "\n";

    std::vector<int> degree_counts(N, 0);
    int max_degree = 0;
    int min_degree = N;

    for (int i = 0; i < N; ++i) {
        int total_degree = outDegree[i] + static_cast<int>(adjListIn[i].size());
        degree_counts[total_degree]++;
        max_degree = std::max(max_degree, total_degree);
        min_degree = std::min(min_degree, total_degree);
    }

    std::cout << "Max degree: " << max_degree << "\n";
    std::cout << "Min degree: " << min_degree << "\n";

    if (N <= 20) {
        std::cout << "\nDegree distribution:\n";
        for (int d = min_degree; d <= max_degree; ++d) {
            if (degree_counts[d] > 0) {
                std::cout << "  Degree " << d << ": " << degree_counts[d] << " nodes\n";
            }
        }
    }

    double max_edges = static_cast<double>(N) * (N - 1);
    double density = getTotalEdges() / max_edges;
    std::cout << "Graph density: " << std::fixed << std::setprecision(4) << density << "\n\n";
}

/// @brief Calculates the total number of edges in the graph.
/// @return Total edge count.
int Graph::getTotalEdges() const {
    int total = 0;
    for (int deg : outDegree) {
        total += deg;
    }
    return total;
}

/// @brief Calculates the average degree of nodes in the graph.
/// @return Average degree (double).
double Graph::getAverageDegree() const {
    if (N == 0) return 0.0;
    return static_cast<double>(getTotalEdges()) / N;
}

/// @brief Clears all edges from the graph while keeping the same number of nodes.
void Graph::clear() {
    for (auto& adj_list : adjListIn) {
        adj_list.clear();
    }
    std::fill(outDegree.begin(), outDegree.end(), 0);
}
