#pragma once

#include <vector>
#include <map>
#include <string>

class Graph {
private:
	std::vector<std::vector<int>> adjacencyMatrix;
	std::map<std::string, int> names;
	int nodeNumber;

public:
	Graph();

	void addVertex();
	void deleteVertex();
	void insertEdge();
	void deleteEdge();

	void generateRandom(int vertices, int edgeChance);

	~Graph();



};