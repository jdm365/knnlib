#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <queue>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../include/distance.h"


template <typename T>
std::vector<std::pair<float, int>> custom_partial_argsort(
		const std::vector<T>& distances,
		const int k
		) {
	std::vector<std::pair<float, int>> mink_pairs(k);
	int   mink_idxs[100];
	float mink_values[100] = {std::numeric_limits<float>::max()};

	for (int idx = 0; idx < (int)distances.size(); ++idx) {
		if (distances[idx] < mink_values[k-1]) {
			int insert_idx = k-1;
			while (insert_idx > 1 && distances[idx] < mink_values[insert_idx-1]) {
				mink_values[insert_idx] = mink_values[insert_idx-1];
				mink_idxs[insert_idx] = mink_idxs[insert_idx-1];
				--insert_idx;
			}
			mink_values[insert_idx] = distances[idx];
			mink_idxs[insert_idx] = idx;
		}
	}

	for (int idx = 0; idx < k; ++idx) {
		mink_pairs[idx] = std::make_pair(mink_values[idx], mink_idxs[idx]);
	}

	return mink_pairs;
}


/*
template <typename T>
std::vector<std::pair<float, int>> custom_partial_argsort(
		std::vector<T>& distances,
		int k
		) {
	std::vector<std::pair<float, int>> mink_pairs(k);
	int   mink_idxs[100];
	float mink_values[100] = {std::numeric_limits<float>::max()};
	float kth_smallest = std::numeric_limits<float>::max();

	for (int idx = 0; idx < (int)distances.size(); ++idx) {
		if (idx < k) {
			mink_values[idx] = distances[idx];
			mink_idxs[idx]   = idx;

			kth_smallest = std::max(kth_smallest, distances[idx]);

			// Sort values and idxs
			for (int jdx = idx; jdx > 0; --jdx) {
				if (mink_values[jdx] < mink_values[jdx - 1]) {
					std::swap(mink_values[jdx], mink_values[jdx - 1]);
					std::swap(mink_idxs[jdx],   mink_idxs[jdx - 1]);
				}
			}
			continue;
		}

		if (distances[idx] < kth_smallest) {
			mink_values[k - 1] = distances[idx];
			mink_idxs[k - 1]   = idx;

			// Sort values and idxs
			for (int jdx = k - 1; jdx > 0; --jdx) {
				if (mink_values[jdx] < mink_values[jdx - 1]) {
					std::swap(mink_values[jdx], mink_values[jdx - 1]);
					std::swap(mink_idxs[jdx],   mink_idxs[jdx - 1]);
				}
			}

			kth_smallest = mink_values[k - 1];
		}
	}

	for (int idx = 0; idx < k; ++idx) {
		mink_pairs[idx] = std::make_pair(mink_values[idx], mink_idxs[idx]);
	}

	return mink_pairs;
}
*/

template <typename T>
// std::vector<std::pair<float, int>> _get_exact_knn(
std::vector<std::pair<float, int>> _get_exact_knn(
	const T* query_vector,
	const std::vector<T>& data,
	std::vector<float>& distances,
	std::vector<int>& indices,
	int dim,
	int k = 5
	) {
	// Calculate distances by dot product
	// query (1, dim) -> query.T (dim, 1)
	// data (n, dim) 
	// distances (n, 1)
	// distances = data @ query.T
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

	// Partial sort distances to get k largest and their indices
	auto start = std::chrono::high_resolution_clock::now();
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
        result.push_back(std::make_pair(2.0f - 2.0f * distances[indices[idx]], indices[idx]));
    }
	*/
	std::vector<std::pair<float, int>> result = custom_partial_argsort(distances, k);

	auto end = std::chrono::high_resolution_clock::now();
	auto partial_sort_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Partial sort time: " << partial_sort_time << std::endl;
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
	std::vector<float> normed_query_vectors(num_queries * dim);
	std::vector<float> normed_data(num_data * dim);

	#pragma omp parallel for if (num_queries > 40)
	for (int query_idx = 0; query_idx < num_queries * dim; ++query_idx) {
		normed_query_vectors[query_idx] = (float)query_vectors[query_idx];
	}
	l2_norm_data(normed_query_vectors, dim);

	#pragma omp parallel for if (num_queries > 40)
	for (int data_idx = 0; data_idx < num_data * dim; ++data_idx) {
		normed_data[data_idx] = (float)data[data_idx];
	}

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
				normed_query_vectors.data() + batch_idx * dim,	// A
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

