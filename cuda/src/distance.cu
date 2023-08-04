#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <chrono>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "distance.h"

#define TPB 256


__global__ void l2_norm_kernel(float *a, int dim, int n) {
	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx < n) {
		float sum = 0.0;
		for (int idx = 0; idx < dim; ++idx) {
			sum += a[idx * dim + idx] * a[idx * dim + idx];
		}
		sum = sqrt(sum);
		for (int idx = 0; idx < dim; ++idx) {
			a[idx * dim + idx] /= sum;
		}
	}
}


__global__ void find_k_smallest_distances(
		float* all_distances,
		float* topk_distances, 
		int* topk_indices,
		int num_data, 
		int current_batch_size, 
		int k
		) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx < current_batch_size) {
		int* rewrite_indices = new int[k];
		float* rewrite_distances = new float[k];

        for (int k_idx = 0; k_idx < k; ++k_idx) {
            float min_value = 100000.0f;
            int min_idx = -1;

            for (int data_idx = 0; data_idx < num_data; ++data_idx) {
                float value = all_distances[data_idx];
                if (value < min_value) {
                    min_value = value;
                    min_idx = data_idx;
                }
            }

            // Set the smallest distance to a large value so it won't be selected again
            all_distances[min_idx] = 1000000.0f;
			rewrite_indices[k_idx] = min_idx;
			rewrite_distances[k_idx] = min_value;

            // Save the result
            topk_indices[query_idx * k + k_idx] = min_idx;
            topk_distances[query_idx * k + k_idx] = 2.0f - 2.0f * min_value;
        }

		for (int k_idx = 0; k_idx < k; ++k_idx) {
			all_distances[rewrite_indices[k_idx]] = rewrite_distances[k_idx];
		}
		delete[] rewrite_indices;
		delete[] rewrite_distances;
    }
}

std::vector<std::vector<std::pair<float, int>>> get_exact_knn_cublas(
		const std::vector<float>& query_vectors,
		const std::vector<float>& data,
		int dim,
		int k
		) {
	const int BATCH_SIZE = std::min(16384, (int)(0.1f * data.size() / dim));
	std::cout << "BATCH_SIZE: " << BATCH_SIZE << std::endl;

	int num_queries = query_vectors.size() / dim;
	int num_data = data.size() / dim;

	float* device_query_vectors;
	float* device_data;
	float* device_distances;
	int* device_topk_indices;
	float* device_topk_distances;

	cudaMalloc((void**)&device_query_vectors, query_vectors.size() * sizeof(float));
	cudaMalloc((void**)&device_data, data.size() * sizeof(float));
	cudaMalloc((void**)&device_distances, BATCH_SIZE * num_data * sizeof(float));
	cudaMalloc((void**)&device_topk_indices, BATCH_SIZE * k * sizeof(int));
	cudaMalloc((void**)&device_topk_distances, BATCH_SIZE * k * sizeof(int));

	cudaMemcpy(device_query_vectors, query_vectors.data(), query_vectors.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

	// L2 norm device_query_vectors and device_data
	l2_norm_kernel<<<(num_queries * num_data + TPB - 1) / TPB, TPB>>>(device_query_vectors, dim, num_queries);
	l2_norm_kernel<<<(num_queries * num_data + TPB - 1) / TPB, TPB>>>(device_data, dim, num_data);
	cudaDeviceSynchronize();

	cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = 1.0;
	float beta  = 0.0;

	std::vector<std::vector<std::pair<float, int>>> results(num_queries);

	std::vector<float> host_indices(num_queries * k);
	std::vector<float> host_distances(num_queries * k);

	auto start = std::chrono::high_resolution_clock::now();
	for (int batch_idx = 0; batch_idx < num_data; batch_idx+=BATCH_SIZE) {
		int current_batch_size = std::min(BATCH_SIZE, num_data - batch_idx);

		cublasSgemm(
				handle, 
				CUBLAS_OP_T, 
				CUBLAS_OP_N, 
				num_data, 
				current_batch_size,
				dim, 
				&alpha, 
				device_data, 
				dim, 
				device_query_vectors, 
				dim, 
				&beta, 
				device_distances, 
				num_data
				);

		find_k_smallest_distances<<<(current_batch_size + TPB - 1) / TPB, TPB>>>(
				device_distances, 
				device_topk_distances,
				device_topk_indices,
				num_data, 
				current_batch_size, 
				k
				);

		cudaMemcpy(
				host_indices.data() + batch_idx * k,
				device_topk_indices,
				current_batch_size * k * sizeof(int), 
				cudaMemcpyDeviceToHost
				);
		cudaMemcpy(
				host_distances.data() + batch_idx * k,
				device_topk_distances,
				current_batch_size * k * sizeof(float), 
				cudaMemcpyDeviceToHost
				);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "KNN time: " << elapsed.count() << std::endl;

	// Build the results
	for (int query_idx = 0; query_idx < num_queries; ++query_idx) {
		for (int k_idx = 0; k_idx < k; ++k_idx) {
			std::cout << host_indices[query_idx * k + k_idx] << " ";
			results[query_idx].push_back(
				std::make_pair(
					host_distances[query_idx * k + k_idx], 
					host_indices[query_idx * k + k_idx]
				)
			);
		}
	}

	cublasDestroy(handle);
	cudaFree(device_query_vectors);
	cudaFree(device_data);
	cudaFree(device_distances);
	cudaFree(device_topk_indices);
	cudaFree(device_topk_distances);

	return results;
}
