#pragma once

#include <cuda_runtime.h>
#include <vector>

__global__ void l2_norm_kernel(float *a, int dim, int n);
__global__ void find_k_smallest_distances(
		float* distances, 
		int* indices,
		int num_data, 
		int current_batch_size, 
		int k,
		int offset 
		);
std::vector<std::vector<std::pair<float, int>>> get_exact_knn_cublas(
		const std::vector<float>& query_vectors,
		const std::vector<float>& data,
		int dim,
		int k
		);
