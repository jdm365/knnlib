#include <iostream>
#include <cblas.h>
#include <queue>
#include <algorithm>
#include <limits>
#include <vector>
#include <chrono>
#include <omp.h>


#include "../include/distance.h"
#include "../include/exact.h"
#include "../include/ivf.h"


void IVFIndex::add(std::vector<float>& data) {
	this->data.insert(this->data.end(), data.begin(), data.end());
	l2_norm_data(this->data, this->dim);
	centroid_assignments.resize((int)data.size() / (int)dim);
}

void IVFIndex::train(std::vector<float>& train_data) {
	// Run k-means
	// Calculate optimal centroids and copy vectors to centroid

	std::cout << "...Training..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	l2_norm_data(train_data, this->dim);

	// Initialize centroids
	#pragma omp parallel for
	for (int idx = 0; idx < num_centroids; ++idx) {
		int rand_idx = rand() % ((int)train_data.size() / (int)dim);
		for (int jdx = 0; jdx < dim; ++jdx) {
			centroids[idx * dim + jdx] = train_data[rand_idx * dim + jdx];
		}
	}

	const int NUM_ITERS = 100;
	float LR = 0.2f;
	const float EPS = 1e-6f;
	std::vector<float> centroid_sums(num_centroids * dim);

	for (int iteration = 0; iteration < NUM_ITERS; ++iteration) {
		std::fill(centroid_sums.begin(), centroid_sums.end(), 0.0f);

		std::vector<std::vector<std::pair<float, int>>> centroid_idxs = get_exact_knn_blas(
				train_data,
				centroids,
				dim,
				1	
				);
		for (int idx = 0; idx < (int)train_data.size() / (int)dim; ++idx) {
			centroid_assignments[idx] = centroid_idxs[idx][0].second;
		}

		// Compute centroid sums to get means and new centroids
		std::vector<int> centroid_counts(num_centroids, 0);

		#pragma omp parallel for schedule(static)
		for (int idx = 0; idx < (int)train_data.size() / (int)dim; ++idx) {
			int centroid_idx = centroid_assignments[idx];
			cblas_saxpy(
					dim,
					1.0f,
					train_data.data() + (idx * dim),
					1,
					centroid_sums.data() + (centroid_idx * dim),
					1
					);
			centroid_counts[centroid_idx]++;
		}

		// Get means and assign to centroids
		double diff_avg = 0.0;
		#pragma omp parallel for schedule(static)
		for (int centroid_idx = 0; centroid_idx < num_centroids; ++centroid_idx) {
			for (int dim_idx = 0; dim_idx < dim; ++dim_idx) {
				centroid_sums[centroid_idx * dim + dim_idx] /= (centroid_counts[centroid_idx] + EPS);
				double diff = centroid_sums[centroid_idx * dim + dim_idx] - centroids[centroid_idx * dim + dim_idx];
				centroids[centroid_idx * dim + dim_idx] += LR * diff;
				diff_avg += std::abs(diff) / (num_centroids * dim);
			}
			// std::cout << "Centroid: " << centroid_idx << " Counts: " << centroid_counts[centroid_idx] << std::endl;
		}
		LR *= 0.9f;
		// std::cout << "Iteration: " << iteration << " Diff: " << diff_avg << std::endl;
		if (diff_avg < 0.005f) {
			break;
		}
	}

	// Copy vectors to centroid_vectors
	// #pragma omp parallel for schedule(static)
	for (int idx = 0; idx < (int)train_data.size() / (int)dim; ++idx) {
		int centroid_idx = centroid_assignments[idx];
		centroid_vectors[centroid_idx].insert(
				centroid_vectors[centroid_idx].end(),
				train_data.begin() + idx * dim,
				train_data.begin() + (idx + 1) * dim
				);
		// Get centroid_indices
		centroid_indices[centroid_idx].push_back(idx);
	}
	// std::cout << "Centroid vectors size: " << centroid_vectors.size() << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Training time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << std::endl;
}


std::vector<std::vector<std::pair<float, int>>> IVFIndex::search(const std::vector<float>& _query, int k) {
	std::vector<float> query = l2_norm_data(_query, dim);

    int num_queries = query.size() / dim;
    std::vector<std::vector<std::pair<float, int>>> final_results(num_queries);

    // Find n_probe nearest centroids
	// auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::pair<float, int>>> nearest_idxs = get_exact_knn_blas(
        query, 
        centroids, 
        dim, 
        n_probe
    );

	// auto end = std::chrono::high_resolution_clock::now();
	// auto nearest_idxs_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	// start = std::chrono::high_resolution_clock::now();

	// Upper bound on size of distances. If all data were in one centroid.
	std::vector<float> distances((int)data.size() / (int)dim);
	std::vector<int>   indices((int)data.size() / (int)dim);
	std::iota(indices.begin(), indices.end(), 0);

    for (int query_idx = 0; query_idx < num_queries; ++query_idx) {
        std::priority_queue<
			std::pair<float, int>, 
			std::vector<std::pair<float, int>>, 
			std::less<>
		> max_heap;
        
        for (int probe_idx = 0; probe_idx < n_probe; ++probe_idx) {
            int centroid_idx = nearest_idxs[query_idx][probe_idx].second;

			if (centroid_vectors[centroid_idx].size() == 0) {
				continue;
			}

			// auto start = std::chrono::high_resolution_clock::now();
			std::vector<std::pair<float, int>> local_results = _get_exact_knn(
					query.data() + query_idx * dim,
					centroid_vectors[centroid_idx],
					distances,
					indices,
					dim,
					std::min(k, (int)centroid_vectors[centroid_idx].size() / dim)
					);
			// auto end = std::chrono::high_resolution_clock::now();
			// auto exact_knn_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            for (auto& result: local_results) {
				// Correct indices with centroid_indices
				result.second = centroid_indices[centroid_idx][result.second];
				if ((int)max_heap.size() < k) {
					max_heap.push(result);
				} 
				else if (result < max_heap.top()) {
					max_heap.pop();
					max_heap.push(result);
				}
        	}
			// std::cout << "Exact knn time: " << exact_knn_time << "us";
			// std::cout << " Centroid count: " << centroid_vectors[centroid_idx].size() / dim << std::endl;
		}

        std::vector<std::pair<float, int>> top_k(k);
        for (int idx = k - 1; idx >= 0; --idx) {
            top_k[idx] = max_heap.top();
            max_heap.pop();
        }
        final_results[query_idx] = top_k;
    }
    return final_results;
}
