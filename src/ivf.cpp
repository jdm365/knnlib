#include <iostream>
#include <cblas.h>
#include <queue>
#include <algorithm>
#include <limits>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>


#include "../include/distance.h"
#include "../include/exact.h"
#include "../include/ivf.h"


void kmeanspp_initialize(
		std::vector<float>& data,
		std::vector<float>& centroids,
		int dim,
		int num_centroids,
		const int kpp_centroids
		) {
	centroids.resize(num_centroids * dim);

	// Initialize centroids using kmeans++ algorithm
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0, (float)data.size() / (float)dim - 1);

	int rand_idx = dis(gen);
	for (int jdx = 0; jdx < dim; ++jdx) {
		centroids[jdx] = data[rand_idx * dim + jdx];
	}

	for (int idx = 1; idx < kpp_centroids; ++idx) {
		std::vector<float> min_dists((int)data.size() / (int)dim, std::numeric_limits<float>::max());

		// Calculate distances to closest centroids
		min_dists = get_min_dists(
				std::vector<float>(data.begin() + idx * dim, data.begin() + (idx + 1) * dim),
				std::vector<float>(centroids.begin(), centroids.begin() + idx * dim),
				dim
				);

		/*
		#pragma omp parallel for
		for (int jdx = 0; jdx < (int)data.size() / dim; ++jdx) {
            for (int kdx = 0; kdx < idx; ++kdx) {
                float dist = distance_l2(
						data.data() + (jdx * dim),
						centroids.data() + (kdx * dim),
						dim
						);
                min_dists[jdx] = std::min(min_dists[jdx], dist);
            }
        }
		*/

		std::discrete_distribution<int> dist(min_dists.begin(), min_dists.end());
		int next_centroid_idx = dist(gen);

		#pragma omp parallel for
		for (int jdx = 0; jdx < dim; ++jdx) {
			centroids[idx * dim + jdx] = data[next_centroid_idx * dim + jdx];
		}
	}

	// Randomly initialize remaining centroids
	#pragma omp parallel for
	for (int idx = kpp_centroids; idx < num_centroids; ++idx) {
		int rand_idx = dis(gen);
		for (int jdx = 0; jdx < dim; ++jdx) {
			centroids[idx * dim + jdx] = data[rand_idx * dim + jdx];
		}
	}
}

void IVFIndex::add(const std::vector<float>& _data) {
	// Add to member varaible data
	l2_norm_data(_data, dim);
	data.insert(data.end(), _data.begin(), _data.end());
	centroid_assignments.resize((int)data.size() / (int)dim);
	size = (int)data.size() / (int)dim;
}

void IVFIndex::train(std::vector<float>& train_data) {
	// Run k-means
	// Calculate optimal centroids and copy vectors to centroid

	std::cout << "...Training..." << std::endl;

	l2_norm_data(train_data, this->dim);

	// Kmeans++ initialization
	const int kpp_centroids = 1;//(int)sqrt((int)sqrt(num_centroids));
	kmeanspp_initialize(train_data, centroids, dim, num_centroids, kpp_centroids);
	
	/*
	// Randomly initialize centroids
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0, (float)train_data.size() / (float)dim - 1);
	for (int idx = 0; idx < num_centroids; ++idx) {
		int rand_idx = dis(gen);
		for (int jdx = 0; jdx < dim; ++jdx) {
			centroids[idx * dim + jdx] = train_data[rand_idx * dim + jdx];
		}
	}
	*/

	const int NUM_ITERS = 10;
	float LR = 0.9f;
	const float EPS = 1e-9f;
	std::vector<float> centroid_sums(num_centroids * dim);
	std::vector<int> centroid_counts(num_centroids, 0);

	auto start = std::chrono::high_resolution_clock::now();
	for (int iteration = 0; iteration < NUM_ITERS; ++iteration) {
		std::fill(centroid_sums.begin(), centroid_sums.end(), 0.0f);

		centroid_assignments = get_centroid_assignments(
				train_data,
				centroids,
				dim
				);

		// Compute centroid sums to get means and new centroids
		std::fill(centroid_counts.begin(), centroid_counts.end(), 0);

		for (int idx = 0; idx < (int)train_data.size() / dim; ++idx) {
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
			#pragma unroll 4
			for (int dim_idx = 0; dim_idx < dim; ++dim_idx) {
				centroid_sums[centroid_idx * dim + dim_idx] /= (centroid_counts[centroid_idx] + EPS);
				double diff = centroid_sums[centroid_idx * dim + dim_idx] - centroids[centroid_idx * dim + dim_idx];
				centroids[centroid_idx * dim + dim_idx] += LR * diff;
				diff_avg += std::abs(diff) / (num_centroids * dim);
			}
		}

		LR *= 0.9f;
		std::cout << "Iteration: " << iteration << " Diff: " << diff_avg << " LR: " << LR << std::endl;
		/*
		if (diff_avg < 0.005f) {
			break;
		}
		*/

		/*
		std::vector<int> sorted_counts = centroid_counts;
		std::sort(sorted_counts.begin(), sorted_counts.end(), std::greater<int>());
		for (int centroid_idx = 0; centroid_idx < num_centroids; ++centroid_idx) {
			std::cout << "Idx: " << centroid_idx << " Count: " << sorted_counts[centroid_idx] << std::endl;
		}
		*/
	}

	// Copy vectors to centroid_vectors
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
	auto end = std::chrono::high_resolution_clock::now();
	auto iteration_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / NUM_ITERS;
	std::cout << "Avg. iteration time: " << iteration_time << " milliseconds" << std::endl;
}


std::vector<std::vector<std::pair<float, int>>> IVFIndex::search(const std::vector<float>& _query, int k) {
	std::vector<float> query = l2_norm_data(_query, dim);

    int num_queries = query.size() / dim;
    std::vector<std::vector<std::pair<float, int>>> final_results(
			num_queries, 
			std::vector<std::pair<float, int>>(k, std::make_pair(0.0f, 0))
			);

    // Find n_probe nearest centroids
    std::vector<std::vector<std::pair<float, int>>> nearest_idxs = get_exact_knn_blas(
        query, 
        centroids, 
        dim, 
        n_probe
    );
	openblas_set_num_threads(8);

	// Upper bound on size of distances. If all data were in one centroid.
	std::vector<float> distances(CHUNK_SIZE);
	std::priority_queue<
		std::pair<float, int>, 
		std::vector<std::pair<float, int>>, 
		std::greater<std::pair<float, int>>
	> max_heap;
    for (int query_idx = 0; query_idx < num_queries; ++query_idx) {

        for (int probe_idx = 0; probe_idx < n_probe; ++probe_idx) {
            int centroid_idx = nearest_idxs[query_idx][probe_idx].second;

			if (centroid_vectors[centroid_idx].size() == 0) {
				continue;
			}

			_get_exact_knn_fused(
					query.data() + query_idx * dim,
					centroid_vectors[centroid_idx],
					distances,
					max_heap,
					centroid_indices[centroid_idx],
					dim,
					std::min(k, (int)centroid_vectors[centroid_idx].size() / dim)
					);
		}

		// Add to final_results by popping from min_heap
		// final_results[query_idx].resize(k);
		for (int idx = 0; idx < std::min(k, (int)max_heap.size()); ++idx) {
			final_results[query_idx][k - idx - 1] = max_heap.top();
			final_results[query_idx][k - idx - 1].first = 2.0f - 2.0f * final_results[query_idx][k - idx - 1].first;
			max_heap.pop();
		}
	}

    return final_results;
}
