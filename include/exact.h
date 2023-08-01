#pragma once

#include <iostream>
#include <vector>
#include <omp.h>

#include "../include/distance.h"


template <typename T>
std::vector<std::pair<float, int>> _get_exact_knn(
	const T* query_vector,
	const std::vector<T>& data,
	int dim,
	int k = 5
	) {
	std::vector<std::pair<float, int>> result;
	result.reserve(k + 1);

	float min_distance_to_keep = std::numeric_limits<float>::max();

	for (int idx = 0; idx < (int)data.size() / (int)dim; ++idx) {
		float distance = distance_l2(
				query_vector,
				data.data() + idx * dim,
				dim
				);

		if (idx < k) {
			result.push_back(std::make_pair(distance, idx));
		}

		else if (distance < min_distance_to_keep) {
			result.push_back(std::make_pair(distance, idx));
			std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
				return b.first > a.first; // Sorting by distance in descending order
			});
			result.pop_back();
			min_distance_to_keep = result.back().first;
		}
	}

	return result;
}


template <typename T>
std::vector<std::vector<std::pair<float, int>>> get_exact_knn(
	const std::vector<T>& query_vectors,
	const std::vector<T>& data,
	int dim,
	int k
) {
	std::vector<std::vector<std::pair<float, int>>> results;
	results.reserve((int)query_vectors.size() / (int)dim);

	#pragma omp parallel for schedule(static)
	for (int idx = 0; idx < (int)query_vectors.size() / (int)dim; ++idx) {
		results[idx] = _get_exact_knn(
				query_vectors.data() + idx * dim,
				data, 
				dim, 
				k
				);
	}

	return results;
}

template <typename T>
std::vector<std::vector<std::pair<float, int>>> get_exact_knn_blas(
    const std::vector<T>& query_vectors,
    const std::vector<T>& data,
    int dim,
    int k
	) {
	// Assume 
	// num_cores * (L1 + L2) + L3 = 32 MB
	// 33_554_432 = 32 * 1024 * 1024
	// 33_554_432 = (dim * sizeof(float) * optimal_batch_size)
	// optimal_batch_size = 33_554_432 / (dim * sizeof(float))
	// const int BATCH_SIZE = 16384;
	const int BATCH_SIZE = std::min((int)(query_vectors.size() * 0.1f / dim), (int)(33554432 / (dim * sizeof(float))));
	std::cout << "BATCH_SIZE: " << BATCH_SIZE << std::endl;

    int num_queries = query_vectors.size() / dim;
	int num_data = data.size() / dim;

	// Apply L2 normalization to all vectors
	std::vector<float> normed_query_vectors(num_queries * dim);
	std::vector<float> normed_data(num_data * dim);

	const float eta = 1e-6f;

	#pragma omp parallel for
	for (int query_idx = 0; query_idx < num_queries * dim; ++query_idx) {
		normed_query_vectors[query_idx] = (float)query_vectors[query_idx];
	}

	#pragma omp parallel for
	for (int query_idx = 0; query_idx < num_queries; ++query_idx) {
		float norm = cblas_snrm2(
				dim,
				normed_query_vectors.data() + query_idx * dim,
				1
				);
		for (int d = 0; d < dim; ++d) {
			normed_query_vectors[query_idx * dim + d] /= (norm + eta);
		}
	}

	#pragma omp parallel for
	for (int data_idx = 0; data_idx < num_data * dim; ++data_idx) {
		normed_data[data_idx] = (float)data[data_idx];
	}

	#pragma omp parallel for
	for (int data_idx = 0; data_idx < num_data; ++data_idx) {
		float norm = cblas_snrm2(
				dim,
				normed_data.data() + data_idx * dim,
				1
				);
		for (int d = 0; d < dim; ++d) {
			normed_data[data_idx * dim + d] /= (norm + eta);
		}
	}

	// Allocate memory for results
	std::vector<std::vector<std::pair<float, int>>> results(num_queries);

	// Allocate memory for distances and indices
	std::vector<float> distances(BATCH_SIZE * num_data);

	for (int batch_idx = 0; batch_idx < num_queries; batch_idx += BATCH_SIZE) {
		int current_batch_size = std::min(BATCH_SIZE, num_queries - batch_idx);

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
				normed_query_vectors.data() + batch_idx * dim,	// A
				dim,									// lda
				normed_data.data(),						// B
				dim,									// ldb
				0.0f,									// beta
				distances.data(),						// C
				num_data								// ldc
				);

		#pragma omp parallel for
        for (int query_idx = 0; query_idx < current_batch_size; ++query_idx) {
            std::vector<std::pair<float, int>> all_distances(num_data);
            
            for (int data_idx = 0; data_idx < num_data; ++data_idx) {
                all_distances[data_idx] = std::make_pair(
						1.0f - distances[query_idx * num_data + data_idx], 
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
