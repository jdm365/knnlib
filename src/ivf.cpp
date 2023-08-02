#include <iostream>
#include <cblas.h>
#include <queue>
#include <algorithm>
#include <limits>
#include <vector>
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

	const int NUM_ITERS = 10;
	std::vector<float> centroid_sums(num_centroids * (int)train_data.size() * dim);

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
			vector_sum(
					centroid_sums.data() + (centroid_idx * dim),
					train_data.data() + (idx * dim),
					centroid_sums.data() + (centroid_idx * dim),
					dim	
					);
			centroid_counts[centroid_idx]++;
		}

		// Get means and assign to centroids
		#pragma omp parallel for schedule(static)
		for (int centroid_idx = 0; centroid_idx < num_centroids; ++centroid_idx) {
			for (int dim_idx = 0; dim_idx < dim; ++dim_idx) {
				centroids[centroid_idx * dim + dim_idx] /= centroid_counts[centroid_idx];
			}
			std::cout << "Centroid: " << centroid_idx << " Counts: " << centroid_counts[centroid_idx] << std::endl;
		}
	}

	// Copy vectors to centroid_vectors
	// #pragma omp parallel for schedule(static)
	for (int idx = 0; idx < (int)train_data.size() / dim; ++idx) {
		int centroid_idx = centroid_assignments[idx];
		centroid_vectors[centroid_idx].insert(
				centroid_vectors[centroid_idx].end(),
				train_data.begin() + idx * dim,
				train_data.begin() + (idx + 1) * dim
				);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Training time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << std::endl;
}


std::vector<std::vector<std::pair<float, int>>> IVFIndex::search(const std::vector<float>& query, int k) {
	std::vector<std::vector<std::vector<std::pair<float, int>>>> results((int)query.size() * n_probe);
	std::vector<std::vector<std::pair<float, int>>> final_results((int)query.size() * k);

	// Find n_probe nearest centroids
	std::vector<std::vector<std::pair<float, int>>> nearest_idxs = get_exact_knn_blas(
			query, 
			centroids, 
			dim, 
			n_probe
			);

	for (int probe_idx = 0; probe_idx < n_probe; ++probe_idx) {
		for (int query_idx = 0; query_idx < (int)query.size(); ++query_idx) {
			int centroid_idx = centroid_assignments[nearest_idxs[query_idx][probe_idx].second];
			std::cout << "Query " << query_idx << " assigned to centroid " << centroid_idx << std::endl;
			results.push_back(get_exact_knn_blas(
						std::vector(
							query.begin() + query_idx * dim, 
							query.begin() + (query_idx + 1) * dim
							),
						centroid_vectors[centroid_idx], 
						dim, 
						k
						));
		}
	}

	// Go through results and exract global min-k distances and indices
	for (int result_idx = 0; result_idx < (int)results.size(); ++result_idx) {
		for (int query_idx = 0; query_idx < (int)query.size(); ++query_idx) {
			for (int k_idx = 0; k_idx < k; ++k_idx) {
				final_results[query_idx * k + k_idx].push_back(
						std::make_pair(
							results[result_idx][query_idx][k_idx].first,
							results[result_idx][query_idx][k_idx].second + result_idx * num_centroids
							)
						);
			}
		}
	}
	return final_results;
}
