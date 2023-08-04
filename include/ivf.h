#pragma once

#include <iostream>
#include <vector>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../include/distance.h"

class IVFIndex {
	public:
		int dim;
		int num_centroids;
		int n_probe;
		std::vector<float>& data = *new std::vector<float>();
		std::vector<float> centroids;
		std::vector<std::vector<float>> centroid_vectors = std::vector<std::vector<float>>(num_centroids);
		std::vector<int> centroid_assignments;

		IVFIndex(
				int dim,
				int num_centroids = 32,
				int n_probe = 1
				): dim(dim), num_centroids(num_centroids), n_probe(n_probe) {
			centroids = std::vector<float>(dim * num_centroids, 0.0f);
		}
		~IVFIndex() { 
			delete &data; 
		}

		void add(std::vector<float>& data);
		void add_wrapper(pybind11::array_t<float> data);
		void train(std::vector<float>& train_data);
		void train_wrapper(pybind11::array_t<float> train_data);
		std::vector<std::vector<std::pair<float, int>>> search(const std::vector<float>& query, int k);
		std::vector<std::vector<std::pair<float, int>>> search_wrapper(pybind11::array_t<float> query, int k);
};
