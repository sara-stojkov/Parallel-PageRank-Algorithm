#pragma once
#include <vector>
#include <map>
#include <string>

class Graph {
public:
    Graph();                 // empty graph
    Graph(int n);            // graph with n nodes
    ~Graph();                // destructor

    int numNodes() const { return N; }

    void addEdge(int from, int to);
    const std::vector<int>& getInNeighbors(int v) const;
    int getOutDegree(int v) const;

private:
    int N;
    std::vector<std::vector<int>> adjListIn;
    std::vector<int> outDegree;
    std::map<std::string, int> names;
};
