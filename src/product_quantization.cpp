#include <iostream>
#include <vector>
#include <cstring>
#include <assert.h>
#include <random>

#include "distance.h"
#include "product_quantization.h"

const int NUM_ITERS = 100;

ProductQuantizer train_quantizer(
		const std::vector<float>& data,
		int dim,
		int num_subvectors,
		int num_centroids
		) {
	assert(dim % num_subvectors == 0);
	int subvector_dim = dim / num_subvectors;

	ProductQuantizer pq;
	pq.num_subvectors = num_subvectors;
	pq.num_centroids  = num_centroids;
	pq.dim            = dim;

	std::vector<uint8_t> centroid_assignments((int)data.size() / subvector_dim);

	// Initialize num_centroids * num_subvectors centroids
	// Centroids stored as ((all_centroids_subspace_0), (all_centroids_subspace_1), ...)
	// std::vector<float> centroids(num_centroids * num_subvectors * subvector_dim);
	pq.centroids.resize(num_centroids * num_subvectors * subvector_dim);
	#pragma omp parallel for
	for (int idx = 0; idx < num_centroids; ++idx) {
		int rand_idx = rand() % ((int)data.size() / (int)dim);
		for (int jdx = 0; jdx < dim; ++jdx) {
			pq.centroids[idx * dim + jdx] = data[rand_idx * dim + jdx];
		}
	}

	// Train kmeans for NUM_ITERS
	std::vector<float> centroid_sums(num_centroids * num_subvectors * subvector_dim);

	for (int iteration = 0; iteration < NUM_ITERS; ++iteration) {
		memset(
				centroid_sums.data(), 
				0.0f, 
				sizeof(float) * num_centroids * num_subvectors * subvector_dim
				);

		float min_distance = std::numeric_limits<float>::max();
		int min_centroid_idx = -1;
		#pragma omp parallel for schedule(static)
		for (int idx = 0; idx < (int)data.size() / (int)dim; ++idx) {
			for (int subvector_idx = 0; subvector_idx < subvector_dim; ++subvector_idx) {
				min_distance = std::numeric_limits<float>::max();
				min_centroid_idx = -1;
				for (int centroid_idx = 0; centroid_idx < num_centroids; ++centroid_idx) {
					float distance = distance_l2(
							data.data() + idx * dim + subvector_idx * subvector_dim,
							pq.centroids.data() + (subvector_dim * num_centroids * subvector_idx) + (centroid_idx * subvector_dim),
							subvector_dim
							);
					if (distance < min_distance) {
						min_distance = distance;
						min_centroid_idx = centroid_idx;
					}
				}
				centroid_assignments[idx * subvector_dim + subvector_idx] = min_centroid_idx;
			}
		}

		// Compute centroid sums to get means and new centroids
		std::vector<int> centroid_counts(num_centroids, 0);

		#pragma omp parallel for schedule(static)
		for (int idx = 0; idx < (int)data.size() / (int)dim; ++idx) {
			for (int subvector_idx = 0; subvector_idx < subvector_dim; ++subvector_idx) {
				int centroid_idx = centroid_assignments[idx * subvector_dim + subvector_idx];
				vector_sum(
						centroid_sums.data() + (subvector_dim * num_centroids * subvector_idx) + (centroid_idx * subvector_dim),
						data.data() + idx * dim + subvector_idx * subvector_dim,
						centroid_sums.data() + (subvector_dim * num_centroids * subvector_idx) + (centroid_idx * subvector_dim),
						subvector_dim
						);
				centroid_counts[centroid_idx]++;
			}
		}
		// Get means and assign to centroids
		#pragma omp parallel for schedule(static)
		for (int centroid_idx = 0; centroid_idx < num_centroids; ++centroid_idx) {
			for (int subvector_idx = 0; subvector_idx < subvector_dim; ++subvector_idx) {
				for (int dim_idx = 0; dim_idx < subvector_dim; ++dim_idx) {
					pq.centroids[centroid_idx * subvector_dim + dim_idx] /= centroid_counts[centroid_idx];
				}
			}
		}
	}
	return pq;
}

void train_quantizer(
		ProductQuantizer& pq,
		const std::vector<float>& data,
		const int dim,
		const int num_subvectors,
		const int num_centroids
		) {
	assert(dim % num_subvectors == 0);
	int subvector_dim = dim / num_subvectors;

	pq.num_subvectors = num_subvectors;
	pq.num_centroids  = num_centroids;
	pq.dim            = dim;

	std::vector<uint8_t> centroid_assignments((int)data.size() / subvector_dim);

	// Initialize num_centroids * num_subvectors centroids
	// Centroids stored as ((all_centroids_subspace_0), (all_centroids_subspace_1), ...)
	pq.centroids.resize(num_centroids * num_subvectors * subvector_dim);
	#pragma omp parallel for
	for (int idx = 0; idx < num_centroids; ++idx) {
		int rand_idx = rand() % ((int)data.size() / (int)dim);
		for (int jdx = 0; jdx < dim; ++jdx) {
			pq.centroids[idx * dim + jdx] = data[rand_idx * dim + jdx];
		}
	}

	// Train kmeans for NUM_ITERS
	std::vector<float> centroid_sums(num_centroids * num_subvectors * subvector_dim);

	for (int iteration = 0; iteration < NUM_ITERS; ++iteration) {
		memset(
				centroid_sums.data(), 
				0.0f, 
				sizeof(float) * num_centroids * num_subvectors * subvector_dim
				);

		float min_distance = std::numeric_limits<float>::max();
		int min_centroid_idx = -1;
		// #pragma omp parallel for schedule(static)
		for (int idx = 0; idx < (int)data.size() / (int)dim; ++idx) {
			for (int subvector_idx = 0; subvector_idx < num_subvectors; ++subvector_idx) {

				min_distance = std::numeric_limits<float>::max();
				min_centroid_idx = -1;

				for (int centroid_idx = 0; centroid_idx < num_centroids; ++centroid_idx) {
					int offset_data = idx * dim + subvector_idx * subvector_dim;
					int offset_centroids = (subvector_dim * num_centroids * subvector_idx) + 
										   (centroid_idx * subvector_dim);
					float distance = distance_l2(
							data.data() + offset_data,
							pq.centroids.data() + offset_centroids,
							subvector_dim
							);
					if (distance < min_distance) {
						min_distance = distance;
						min_centroid_idx = centroid_idx;
					}
				}
				centroid_assignments[idx * num_subvectors + subvector_idx] = min_centroid_idx;
			}
		}

		// Compute centroid sums to get means and new centroids
		std::vector<int> centroid_counts(num_centroids, 0);

		// #pragma omp parallel for schedule(static)
		for (int idx = 0; idx < (int)data.size() / (int)dim; ++idx) {
			for (int subvector_idx = 0; subvector_idx < num_subvectors; ++subvector_idx) {
				int centroid_idx = centroid_assignments[idx * num_subvectors + subvector_idx];

				int offset_data 	 = idx * dim + subvector_idx * subvector_dim;
				int offset_centroids = (subvector_dim * num_centroids * subvector_idx) + 
									   (centroid_idx * subvector_dim);

				vector_sum(
						centroid_sums.data() + offset_centroids,
						data.data() + offset_data,
						centroid_sums.data() + offset_centroids,
						subvector_dim
						);
				centroid_counts[centroid_idx]++;
			}
		}
		for (int centroid_idx = 0; centroid_idx < num_centroids; ++centroid_idx) {
			std::cout << "Centroid " << centroid_idx << " count: " << centroid_counts[centroid_idx] << std::endl;
		}

		// Get means and assign to centroids
		// #pragma omp parallel for schedule(static)
		for (int centroid_idx = 0; centroid_idx < num_centroids; ++centroid_idx) {
			for (int subvector_idx = 0; subvector_idx < num_subvectors; ++subvector_idx) {
				for (int dim_idx = 0; dim_idx < subvector_dim; ++dim_idx) {
					int idx = (subvector_dim * num_centroids * subvector_idx) + 
							  (centroid_idx * subvector_dim) + dim_idx;
					pq.centroids[idx] /= centroid_counts[centroid_idx];
				}
			}
		}
	}
}

std::vector<uint8_t> product_quantize(
		const std::vector<float>& data,
		const ProductQuantizer& quantizer
		) {
	assert(data.size() % quantizer.dim == 0);

	int num_vectors   = data.size() / quantizer.dim;
	int subvector_dim = quantizer.dim / quantizer.num_subvectors;

	std::vector<uint8_t> quantized_data(num_vectors * quantizer.num_subvectors);


	#pragma omp parallel for
	for (int idx = 0; idx < num_vectors; ++idx) {
		for (int subvector_idx = 0; subvector_idx < quantizer.num_subvectors; ++subvector_idx) {
			float min_distance = std::numeric_limits<float>::max();
			int   min_centroid_idx = -1;
			for (int centroid_idx = 0; centroid_idx < quantizer.num_centroids; ++centroid_idx) {
				int offset_data = idx * quantizer.dim + subvector_idx * subvector_dim;
				int offset_centroids = (subvector_dim * quantizer.num_centroids * subvector_idx) + 
									   (centroid_idx * subvector_dim);
				float distance = distance_l2(
						data.data() + offset_data,
						quantizer.centroids.data() + offset_centroids,
						subvector_dim
						);
				if (distance < min_distance) {
					min_distance = distance;
					min_centroid_idx = centroid_idx;
				}
			}
			quantized_data[idx * quantizer.num_subvectors + subvector_idx] = min_centroid_idx;
		}
	}

	return quantized_data;
}
