#include "Graph.h"

// Empty constructor
Graph::Graph()
    : N(0), adjListIn(), outDegree(), names() {
}

// Constructor with n nodes
Graph::Graph(int n)
    : N(n), adjListIn(n), outDegree(n, 0), names() {
}

// Destructor
Graph::~Graph() {}

// Add a directed edge from -> to
void Graph::addEdge(int from, int to) {
    if (from < 0 || from >= N || to < 0 || to >= N) return;
    adjListIn[to].push_back(from);
    outDegree[from]++;
}

const std::vector<int>& Graph::getInNeighbors(int v) const {
    return adjListIn[v];
}

int Graph::getOutDegree(int v) const {
    return outDegree[v];
}
