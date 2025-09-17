// Sara Stojkov SV38/2023

#include <iostream>
#include "Graph.h"
#include "PageRank.h"



void testEmptyGraph() {
    Graph g;
    std::cout << "[Test Empty] numNodes = " << g.numNodes() << " (expected 0)" << std::endl;
}

void testSmallGraph() {
    Graph g(3);
    g.addEdge(0, 1);
    g.addEdge(1, 2);
    g.addEdge(2, 0);

    std::cout << "[Test Small] numNodes = " << g.numNodes() << " (expected 3)" << std::endl;

    // Out-degrees
    for (int i = 0; i < g.numNodes(); i++) {
        std::cout << "Out-degree(" << i << ") = " << g.getOutDegree(i) << std::endl;
    }

    // In-neighbors
    for (int i = 0; i < g.numNodes(); i++) {
        std::cout << "In-neighbors(" << i << "): ";
        for (int neighbor : g.getInNeighbors(i)) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }
}

void testSelfLoop() {
    Graph g(2);
    g.addEdge(0, 0); // self-loop
    g.addEdge(1, 0);

    std::cout << "[Test Self-loop]" << std::endl;
    std::cout << "Out-degree(0) = " << g.getOutDegree(0) << " (expected 1)" << std::endl;
    std::cout << "Out-degree(1) = " << g.getOutDegree(1) << " (expected 1)" << std::endl;

    std::cout << "In-neighbors(0): ";
    for (int n : g.getInNeighbors(0)) std::cout << n << " ";
    std::cout << std::endl;
}

void testInvalidEdge() {
    Graph g(3);
    g.addEdge(0, 5); // invalid, ignored
    g.addEdge(-1, 1); // invalid, ignored

    std::cout << "[Test Invalid Edge] numNodes = " << g.numNodes() << " (expected 3)" << std::endl;
    for (int i = 0; i < g.numNodes(); i++) {
        std::cout << "Out-degree(" << i << ") = " << g.getOutDegree(i) << std::endl;
    }
}
int main()
{
    testEmptyGraph();
    testSmallGraph();
    testSelfLoop();
    testInvalidEdge();


    std::cout << "PageRank Algorithm!\n";
}