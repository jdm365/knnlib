#pragma once

#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <algorithm>
#include <list>
#include <queue>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../include/ivf.h"
#include "../include/distance.h"
#include "../include/sort.h"

template <typename T>
std::vector<std::pair<float, int>> _get_exact_knn_fused(
	const T* query_vector,
	const std::vector<T>& data,
	std::vector<float>& chunk_distances,
	int dim,
	int k = 5
	) {
	// Calculate distances by dot product
	// query (1, dim) -> query.T (dim, 1)
	// data (n, dim) 
	// distances (n, 1)
	// distances = data @ query.T

	// auto start = std::chrono::high_resolution_clock::now();
	std::priority_queue<
		std::pair<float, int>,
		std::vector<std::pair<float, int>>,
		std::greater<std::pair<float, int>>
	> max_heap;

	for (int chunk_start = 0; chunk_start < (int)data.size() / dim; chunk_start += CHUNK_SIZE) {
		int chunk_rows = std::min(CHUNK_SIZE, ((int)data.size() / dim) - chunk_start);
		
		cblas_sgemv(
			CblasRowMajor,
			CblasNoTrans,
			chunk_rows,
			dim,
			1.0f,
			data.data() + chunk_start * dim,
			dim,
			query_vector,
			1,
			0.0f,
			chunk_distances.data(),
			1
		);

		for (int idx = 0; idx < chunk_rows; ++idx) {
			float distance = chunk_distances[idx];
			if ((int)max_heap.size() < k) {
				max_heap.push({distance, chunk_start + idx});
				continue;
			}
			if (distance > max_heap.top().first) [[unlikely]] {
				max_heap.pop();
				max_heap.push({distance, chunk_start + idx});
			}
		}
	}

	// Get results from max_heap
	std::vector<std::pair<float, int>> result;
	result.reserve(k);
	while (!max_heap.empty()) {
		result.emplace_back(2.0f - 2.0f * max_heap.top().first, max_heap.top().second);
		max_heap.pop();
	}
	// auto end = std::chrono::high_resolution_clock::now();
	// auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	// std::cout << "Search time: " << search_time << std::endl << std::endl;

	return result;
}


template <typename T>
std::vector<std::pair<float, int>> _get_exact_knn(
	const T* query_vector,
	const std::vector<T>& data,
	std::vector<float>& distances,
	int dim,
	float& threshold,
	int k = 5
	) {
	// Calculate distances by dot product
	// query (1, dim) -> query.T (dim, 1)
	// data (n, dim) 
	// distances (n, 1)
	// distances = data @ query.T
	// auto start = std::chrono::high_resolution_clock::now();
	cblas_sgemv(
			CblasRowMajor, // assuming data is stored in row-major order
			CblasNoTrans,  // no transpose of data
			data.size() / dim, // number of rows
			dim,          // number of columns
			1.0f,         // alpha
			data.data(),  // data matrix
			dim,          // leading dimension of data
			query_vector, // query vector
			1,            // increment for the elements of query_vector
			0.0f,         // beta
			distances.data(), // result vector
			1             // increment for the elements of result vector
		);
	// auto end = std::chrono::high_resolution_clock::now();
	// auto gemv_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// std::vector<int> indices(distances.size());
	// std::iota(indices.begin(), indices.end(), 0);

	// start = std::chrono::high_resolution_clock::now();
	// Partial sort distances to get k largest and their indices
	/*
	std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
        [&distances](int a, int b) {
            return distances[a] > distances[b];
        }
    );

    // Sort the top k elements
    std::sort(indices.begin(), indices.begin() + k,
        [&distances](int a, int b) {
            return distances[a] > distances[b];
        }
    );

	std::vector<std::pair<float, int>> result;
	result.reserve(k);
	for (int idx = 0; idx < k; ++idx) {
		result.emplace_back(2.0f - 2.0f * distances[indices[idx]], indices[idx]);
	}
	*/
	std::vector<int> idxs = get_smallest_k_elements(distances, k, threshold);
	if (idxs.size() == 0) {
		// end = std::chrono::high_resolution_clock::now();

		// auto partial_sort_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		// std::cout << "gemv time: " << gemv_time << std::endl;
		// std::cout << "Partial sort time: " << partial_sort_time << std::endl << std::endl;

		return std::vector<std::pair<float, int>>();
	}

    std::vector<std::pair<float, int>> result;
    result.reserve(k);
	for (int idx = 0; idx < std::min(k, (int)idxs.size()); ++idx) {
		result.emplace_back(2.0f - 2.0f * distances[idxs[idx]], idxs[idx]);
	}
	// end = std::chrono::high_resolution_clock::now();

	// auto partial_sort_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	// std::cout << "gemv time: " << gemv_time << std::endl;
	// std::cout << "Partial sort time: " << partial_sort_time << std::endl << std::endl;
	return result;
}


template <typename T>
std::vector<std::vector<std::pair<float, int>>> get_exact_knn_blas(
    const std::vector<T>& query_vectors,
    const std::vector<T>& data,
    int dim,
    int k
	) {
    long num_queries = query_vectors.size() / dim;
	long num_data = data.size() / dim;

	// Assume 
	// num_cores * (L1 + L2) + L3 ~= 32 MB
	// 33_554_432 = 32 * 1024 * 1024
	// 33_554_432 = (dim * sizeof(float) * optimal_batch_size)
	// optimal_batch_size = 33_554_432 / (dim * sizeof(float))
	// const int BATCH_SIZE = 16384;
	int BATCH_SIZE = std::min(
			(int)(query_vectors.size() * 0.1f / dim), 
			(int)(33554432 / (dim * sizeof(float)))
			);
	if (BATCH_SIZE < 1024) {
		BATCH_SIZE = (int)num_queries;
	}

	// Apply L2 normalization to all vectors
	std::vector<float> normed_data = l2_norm_data(data, dim);

	// Allocate memory for results
	std::vector<std::vector<std::pair<float, int>>> results(num_queries);

	// Allocate memory for distances and indices
	std::vector<float> distances(BATCH_SIZE * num_data);

	for (int batch_idx = 0; batch_idx < num_queries; batch_idx += BATCH_SIZE) {
		int current_batch_size = std::min((long)BATCH_SIZE, num_queries - batch_idx);

		// Get distances with Blas SGEMM
		// distances = query_batch * data^T
		// query_batch: (current_batch_size, dim)
		// data: (num_data, dim)
		// distances: (current_batch_size, num_data)
		cblas_sgemm(
				CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				current_batch_size,						// M
				num_data,								// N
				dim,									// K
				1.0f,									// alpha
				query_vectors.data() + batch_idx * dim,	// A
				dim,									// lda
				normed_data.data(),						// B
				dim,									// ldb
				0.0f,									// beta
				distances.data(),						// C
				num_data								// ldc
				);

		#pragma omp parallel for if (num_queries > 40)
        for (int query_idx = 0; query_idx < current_batch_size; ++query_idx) {
            std::vector<std::pair<float, int>> all_distances(num_data);
            
            for (int data_idx = 0; data_idx < num_data; ++data_idx) {
                all_distances[data_idx] = std::make_pair(
						2.0f - 2.0f * distances[query_idx * num_data + data_idx], 
						data_idx
						);
            }

            // Partially sort to get the k smallest distances
            std::partial_sort(all_distances.begin(), all_distances.begin() + k, all_distances.end());
            results[batch_idx + query_idx].assign(all_distances.begin(), all_distances.begin() + k);
        }
    }

	return results;
}



template <typename T>
std::vector<int> get_centroid_assignments(
    const std::vector<T>& query_vectors,
    const std::vector<T>& _data,
    int dim
	) {
    long num_queries = query_vectors.size() / dim;
	long num_data = _data.size() / dim;

	const int BATCH_SIZE = 1024;

	std::vector<T> data = l2_norm_data(_data, dim);

	// Allocate memory for results
	std::vector<int> results(num_queries);

	// Allocate memory for distances and indices
	std::vector<float> distances(BATCH_SIZE * num_data);

	for (int batch_idx = 0; batch_idx < num_queries; batch_idx += BATCH_SIZE) {
		int current_batch_size = std::min((long)BATCH_SIZE, num_queries - batch_idx);

		// Get distances with Blas SGEMM
		// distances = query_batch * data^T
		// query_batch: (current_batch_size, dim)
		// data: (num_data, dim)
		// distances: (current_batch_size, num_data)
		cblas_sgemm(
				CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				current_batch_size,						// M
				num_data,								// N
				dim,									// K
				1.0f,									// alpha
				query_vectors.data() + batch_idx * dim,	// A
				dim,									// lda
				data.data(),							// B
				dim,									// ldb
				0.0f,									// beta
				distances.data(),						// C
				num_data								// ldc
				);

		// Get argmins of distances.
		// #pragma omp parallel for schedule(static)
		for (int query_idx = 0; query_idx < current_batch_size; ++query_idx) {
			float max_dist = -1.0f;
			int   max_idx  = -1;
			for (int dist_idx = 0; dist_idx < num_data; ++dist_idx) {
				if (distances[query_idx * num_data + dist_idx] > max_dist) {
					max_dist = distances[query_idx * num_data + dist_idx];
					max_idx = dist_idx;
				}
			}
			results[batch_idx + query_idx] = max_idx;
			/*
			results[batch_idx + query_idx] = argmax(
					distances.data() + query_idx * num_data, 
					(int)num_data
					);
			*/
		}
    }

	return results;
}

template <typename T>
std::vector<float> get_min_dists(
    const std::vector<T>& query_vectors,
    const std::vector<T>& _data,
    int dim
	) {
    long num_queries = query_vectors.size() / dim;
	long num_data = _data.size() / dim;

	const int BATCH_SIZE = 1024;

	std::vector<T> data = l2_norm_data(_data, dim);

	// Allocate memory for results
	std::vector<float> results(num_queries);

	// Allocate memory for distances and indices
	std::vector<float> distances(BATCH_SIZE * num_data);

	for (int batch_idx = 0; batch_idx < num_queries; batch_idx += BATCH_SIZE) {
		int current_batch_size = std::min((long)BATCH_SIZE, num_queries - batch_idx);

		// Get distances with Blas SGEMM
		// distances = query_batch * data^T
		// query_batch: (current_batch_size, dim)
		// data: (num_data, dim)
		// distances: (current_batch_size, num_data)
		cblas_sgemm(
				CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				current_batch_size,						// M
				num_data,								// N
				dim,									// K
				1.0f,									// alpha
				query_vectors.data() + batch_idx * dim,	// A
				dim,									// lda
				data.data(),							// B
				dim,									// ldb
				0.0f,									// beta
				distances.data(),						// C
				num_data								// ldc
				);

		// Get max distances.
		#pragma omp parallel for schedule(static)
		for (int query_idx = 0; query_idx < current_batch_size; ++query_idx) {
			float max_dist = -1.0f;
			for (int dist_idx = 0; dist_idx < num_data; ++dist_idx) {
				if (distances[query_idx * num_data + dist_idx] > max_dist) {
					max_dist = distances[query_idx * num_data + dist_idx];
				}
			}
			results[batch_idx + query_idx] = 2.0f - 2.0f * max_dist;
		}
    }

	return results;
}

class FlatIndexL2 {
	public:
		std::vector<float>& data = *new std::vector<float>();
		int dim;

		FlatIndexL2(int dim): dim(dim) {}
		~FlatIndexL2() { delete &data; }

		void add(std::vector<float>& data);
		void add_wrapper(pybind11::array_t<float> data);
		void train(std::vector<float>& data [[maybe_unused]]) {};
		void train_wrapper(pybind11::array_t<float> data [[maybe_unused]]) {};
		std::vector<std::vector<std::pair<float, int>>> search(const std::vector<float>& query, int k);
		std::vector<std::vector<std::pair<float, int>>> search_wrapper(pybind11::array_t<float> query, int k);
};

