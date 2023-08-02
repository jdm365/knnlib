#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "distance.h"


int main() {
	const long DIM = 512;
	const long N   = 1000 * 100;

	std::vector<float> data;
	data.reserve(N * DIM);

	std::mt19937 gen(std::random_device{}()); // Mersenne Twister engine
	// Set seed
	gen.seed(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (long idx = 0; idx < N * DIM; ++idx) {
		data.push_back(dist(gen));
	}

	auto start = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<std::pair<float, int>>> results = get_exact_knn_cublas(data, data, DIM, 5);
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time: ";
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << " ms" << std::endl;

	std::cout << "Nearest neighbors: \n";
	std::cout << "(distance: " << results[500][0].first << ", index: " << results[500][0].second << ")" << std::endl;
	std::cout << "(distance: " << results[500][1].first << ", index: " << results[500][1].second << ")" << std::endl;
	std::cout << "(distance: " << results[500][2].first << ", index: " << results[500][2].second << ")" << std::endl;
	std::cout << "(distance: " << results[500][3].first << ", index: " << results[500][3].second << ")" << std::endl;
	std::cout << "(distance: " << results[500][4].first << ", index: " << results[500][4].second << ")" << std::endl;

	return 0;
}
