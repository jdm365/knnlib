#pragma once

#include <iostream>
#include <vector>
#include <omp.h>

#include "../include/distance.h"

class IVFIndex {
	public:
		int dim;
		int num_centroids;
		int n_probe;
		std::vector<float>& data = *new std::vector<float>();
		std::vector<float> centroids;
		std::vector<float>* centroid_vectors = new std::vector<float>[num_centroids];
		std::vector<int> centroid_assignments;

		IVFIndex(
				int dim,
				int num_centroids = 32,
				int n_probe = 1
				): dim(dim), num_centroids(num_centroids), n_probe(n_probe) {
			centroids = std::vector<float>(dim * num_centroids, 0.0f);
		}
		~IVFIndex() { 
			delete[] centroid_vectors;
			delete &data; 
		}

		void add(std::vector<float>& data);
		void train(std::vector<float>& train_data);
		std::vector<std::vector<std::pair<float, int>>> search(const std::vector<float>& query, int k);
};
