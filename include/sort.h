#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <limits>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../include/distance.h"


template <typename T>
std::vector<int> get_smallest_k_elements(
		const std::vector<T>& distances, int k,
		float& threshold
		) {
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<float> keep_distances;
	std::vector<int>   keep_indices;
	keep_distances.reserve(1000);
	keep_indices.reserve(1000);

	// std::cout << "Threshold: " << threshold << std::endl;
	#pragma omp parallel for schedule(static)
	for (int idx = 0; idx < (int)distances.size(); ++idx) {
		if (distances[idx] > threshold) [[unlikely]] {
			keep_distances.push_back(distances[idx]);
			keep_indices.push_back(idx);

			if ((int)keep_distances.size() == k) {
				// Set threshold to max of keep_distances
				threshold = std::max(*std::min_element(keep_distances.begin(), keep_distances.end()), threshold);
			}
			else if ((int)keep_distances.size() < k) {
				continue;
			}
			else {
				threshold = distances[idx];
			}
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto filter_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	std::cout << "Size of keep_distances: " << keep_distances.size() << std::endl;
	if ((int)keep_distances.size() == 0) {
		std::cout << "Filter time: " << filter_time << std::endl;
		return std::vector<int>();
	}
	else if ((int)keep_distances.size() < k) {
		std::cout << "Filter time: " << filter_time << std::endl;
		return keep_indices;
	}
	start = std::chrono::high_resolution_clock::now();
	std::partial_sort(keep_indices.begin(), keep_indices.begin() + k, keep_indices.end(),
		[&keep_distances](int a, int b) {
			return keep_distances[a] > keep_distances[b];
		}
	);
	end = std::chrono::high_resolution_clock::now();
	auto partial_sort_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Filter time: " << filter_time << std::endl;
	std::cout << "Partial sort time: " << partial_sort_time << std::endl;
	return keep_indices;
}


template <typename T>
void pair_sort(std::vector<T>& distances, std::vector<int>& indices) {
	// From large to small
	std::sort(indices.begin(), indices.end(),
		[&distances](int a, int b) {
			return distances[a] > distances[b];
		}
	);
}

int argmax(float *a, int n);
