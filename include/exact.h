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
#include "../include/avx2.h"


void _get_exact_knn_fused_avx2(
	const float* query_vector,
	const std::vector<float>& data,
	std::priority_queue<
		std::pair<float, int>, 
		std::vector<std::pair<float, int>>, 
		std::greater<std::pair<float, int>>
	>& max_heap,
	std::vector<int>& index_map,
	int dim
	);

void _get_exact_knn_fused(
	const float* query_vector,
	const std::vector<float>& data,
	std::vector<float>& chunk_distances,
	std::priority_queue<
		std::pair<float, int>, 
		std::vector<std::pair<float, int>>, 
		std::greater<std::pair<float, int>>
	>& max_heap,
	std::vector<int>& index_map,
	int dim,
	int k = 5
	);

std::vector<std::pair<float, int>> _get_exact_knn(
	const float* query_vector,
	const std::vector<float>& data,
	std::vector<float>& distances,
	int dim,
	float& threshold,
	int k = 5
	);

std::vector<std::vector<std::pair<float, int>>> get_exact_knn_blas(
    const std::vector<float>& query_vectors,
    const std::vector<float>& data,
    int dim,
    int k
	);

std::vector<int> get_centroid_assignments(
    const std::vector<float>& query_vectors,
    const std::vector<float>& _centroids,
    int dim
	);

std::vector<float> get_min_dists(
    const std::vector<float>& query_vectors,
    const std::vector<float>& _data,
    int dim
	);

void _get_exact_knn_pq(
	const uint8_t* query_vector,
	const std::vector<uint8_t>& data,
	std::vector<float>& chunk_distances,
	std::priority_queue<
		std::pair<float, int>, 
		std::vector<std::pair<float, int>>, 
		std::greater<std::pair<float, int>>
	>& max_heap,
	std::vector<int>& index_map,
	int dim,
	int k = 5
	);

void uint8_vector_matrix_dot_product(
	const uint8_t* query_vectors,
	const uint8_t* data,
	float* distances,
	int dim,
	int num_data
	);



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
