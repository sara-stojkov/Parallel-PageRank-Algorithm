#pragma once

#include <string>
#include <vector>
#include <iostream>

namespace Utils {
 
	static void writeToFile(const std::string filename, const std::vector<double>& ranks);
	static void printTopN(const std::vector<double>& ranks, int N = 10);
	static double getTime();
};