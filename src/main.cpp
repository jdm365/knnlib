#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

#include "../include/exact.h"
#include "../include/product_quantization.h"
#include "../include/ivf.h"


int main() {
	const long DIM = 128;
	const long INDEX_SIZE = 1024 * 1024;
	const long QUERY_SIZE = 1024;
	const int NUM_CENTROIDS = 32;
	const int N_PROBE = 4;

	std::vector<float> data(INDEX_SIZE * DIM);

	std::mt19937 gen(std::random_device{}()); // Mersenne Twister engine
	// Set seed
	gen.seed(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (long idx = 0; idx < INDEX_SIZE * DIM; ++idx) {
		data[idx] = dist(gen);
	}

	// FlatIndexL2 index(DIM);
	IVFIndex index(DIM, NUM_CENTROIDS, N_PROBE);
	index.add(data);
	index.train(data);

	/*
	int num_subvectors = 32;
	int num_centroids = 32;
	std::vector<uint8_t> quantized_data = product_quantize(data, DIM, num_subvectors, num_centroids);

	std::cout << "Original data size:  " << data.size() * sizeof(float) / 1048576 << " MB" << std::endl;
	std::cout << "Quantized data size: " << quantized_data.size() * sizeof(uint8_t) / 1048576 << " MB" << std::endl;

	int quantized_dim = DIM / num_subvectors;
	std::vector<uint8_t> quantized_query_data = std::vector<uint8_t>(quantized_data.begin(), quantized_data.begin() + QUERY_SIZE * DIM / 8);
	*/
	std::vector<float> query_data = std::vector<float>(data.begin(), data.begin() + QUERY_SIZE * DIM);

	auto start = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<std::pair<float, int>>> results = index.search(query_data, 5);
	/*
	std::vector<std::vector<std::pair<float, int>>> results = get_exact_knn_blas(
			quantized_query_data, 
			quantized_data, 
			quantized_dim, 
			5
			);
	*/
	/*
	std::vector<std::vector<std::pair<float, int>>> results = get_exact_knn(
			query_data, 
			quantized_data, 
			quantized_dim, 
			5
			);
	*/
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time: ";
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << " ms" << std::endl;

	std::cout << "Nearest neighbors: \n";
	std::cout << "(distance: " << results[5][0].first << ", index: " << results[5][0].second << ")" << std::endl;
	std::cout << "(distance: " << results[5][1].first << ", index: " << results[5][1].second << ")" << std::endl;
	std::cout << "(distance: " << results[5][2].first << ", index: " << results[5][2].second << ")" << std::endl;
	std::cout << "(distance: " << results[5][3].first << ", index: " << results[5][3].second << ")" << std::endl;
	std::cout << "(distance: " << results[5][4].first << ", index: " << results[5][4].second << ")" << std::endl;

	return 0;
}
