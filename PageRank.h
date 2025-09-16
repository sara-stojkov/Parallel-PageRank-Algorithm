#pragma once

#include "Graph.h"
#include <vector>

class PageRank {

public:
	PageRank(const Graph& g, double d = 0.85);

	void runSequential(int maxItr, double epsilon);
	void runParallel(int maxIter, double epsilon, int numThreads);

	const std::vector<double>& getRanks() const;

private:
	const Graph& graph;
	double damping;
	std::vector<double> prOld, prNew;


};