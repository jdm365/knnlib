#pragma once

#include <iostream>
#include <vector>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../include/distance.h"

#define CHUNK_SIZE 256

class IVFIndex {
	public:
		int dim;
		int num_centroids;
		int n_probe;
		int size = 0;
		alignas(16) std::vector<float> data;
		alignas(16) std::vector<float> centroids;
		alignas(16) std::vector<std::vector<float>> centroid_vectors = std::vector<std::vector<float>>(num_centroids);
		std::vector<int> centroid_assignments;
		std::vector<std::vector<int>> centroid_indices = std::vector<std::vector<int>>(num_centroids);

		IVFIndex(
				int dim,
				int num_centroids = 32,
				int n_probe = 1
				): dim(dim), num_centroids(num_centroids), n_probe(n_probe) {
			centroids = std::vector<float>(dim * num_centroids, 0.0f);
		}
		~IVFIndex() {}

		void add(const std::vector<float>& _data);
		void add_wrapper(pybind11::array_t<float> _data);
		void train(std::vector<float>& train_data);
		void train_wrapper(pybind11::array_t<float> train_data);
		std::vector<std::vector<std::pair<float, int>>> search(const std::vector<float>& query, int k);
		std::vector<std::vector<std::pair<float, int>>> search_wrapper(pybind11::array_t<float> query, int k);
		std::vector<std::vector<std::pair<float, int>>> search_auto(const std::vector<float>& query, int k);
};


void kmeanspp_initialize(
		std::vector<float>& data,
		std::vector<float>& centroids,
		int dim,
		int num_centroids,
		const int kpp_centroids
		);
void kmeans(
		std::vector<float>& data,
		std::vector<float>& centroids,
		std::vector<int>& centroid_assignments,
		int dim,
		int num_centroids,
		int num_iters,
		float lr
		);
