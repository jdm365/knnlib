#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <cblas.h>

#include "exact.h"
#include "product_quantization.h"
#include "ivf.h"



void bench_u8_dot_product() {
	/*************************************************************
	 * Benchmark the function uint8_vector_matrix_dot_product on random vectors
	 * Measure GFLOPS and log to stdout
	 *************************************************************/

	const int N = 100000;
	const int DIM = 512;

	std::mt19937 gen(std::random_device{}());
	std::uniform_int_distribution<uint8_t> dist(0, 255);

	// Align to 64 bytes
	std::vector<uint8_t> vectors(DIM);
	std::vector<uint8_t> matrix(N * DIM);
	std::vector<float> distances(N);

	for (int idx = 0; idx < DIM; ++idx) {
		vectors[idx] = dist(gen);
	}

	for (int idx = 0; idx < N * DIM; ++idx) {
		matrix[idx] = dist(gen);
	}

	auto start = std::chrono::high_resolution_clock::now();
	uint8_vector_matrix_dot_product(
			vectors.data(), 
			matrix.data(), 
			distances.data(),
			DIM,
			N
			);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	double gflops = (double) DIM * N * 2 / duration;
	gflops /= 1000;
	std::cout << "GFLOPS: " << gflops << std::endl;
}

void bench_blas_dot_product() {
	/*************************************************************
	 * Benchmark the function uint8_vector_matrix_dot_product on random vectors
	 * Measure GFLOPS and log to stdout
	 *************************************************************/

	const int N = 100000;
	const int DIM = 512;

	std::mt19937 gen(std::random_device{}());
	std::uniform_real_distribution<float> dist(0, 1);

	std::vector<float> vectors(DIM);
	std::vector<float> matrix(N * DIM);
	std::vector<float> distances(N);

	for (int idx = 0; idx < DIM; ++idx) {
		vectors[idx] = dist(gen);
	}

	for (int idx = 0; idx < N * DIM; ++idx) {
		matrix[idx] = dist(gen);
	}

	auto start = std::chrono::high_resolution_clock::now();
	// Use cblas_sgemv to compute the dot product
	cblas_sgemv(
			CblasRowMajor,
			CblasNoTrans,
			N,
			DIM,
			1.0,
			matrix.data(),
			DIM,
			vectors.data(),
			1,
			0.0,
			distances.data(),
			1
			);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	double gflops = (double) DIM * N * 2 / duration;
	gflops /= 1000;
	std::cout << "GFLOPS: " << gflops << std::endl;
}

void test_ivfpq() {
	/*************************************************************
	 * Measure GFLOPS and log to stdout
	 *************************************************************/

	const int N_QUERIES = 1000;
	const int N = 100000;
	const int DIM = 128;
	const int NUM_CENTROIDS = 64;

	std::mt19937 gen(std::random_device{}());
	std::uniform_real_distribution<float> dist(0, 1);

	std::vector<float> vectors(DIM);
	std::vector<float> matrix(N * DIM);
	std::vector<float> distances(N);

	for (int idx = 0; idx < DIM; ++idx) {
		vectors[idx] = dist(gen);
	}

	for (int idx = 0; idx < N * DIM; ++idx) {
		matrix[idx] = dist(gen);
	}

	IVFIndex ivf_index(DIM, NUM_CENTROIDS, true);
	ivf_index.train(matrix);
	ivf_index.add(matrix);
	ivf_index.search(vectors, 5);

	auto start = std::chrono::high_resolution_clock::now();

	for (int idx = 0; idx < N_QUERIES; ++idx) {
		ivf_index.search(vectors, 5);
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "Time: " << duration << std::endl;
}

int main() {
	// bench_u8_dot_product();
	// bench_blas_dot_product();
	test_ivfpq();
	return 0;
}
