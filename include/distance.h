#pragma once

#include <vector>
#include <cblas.h>

template <typename T>
inline float distance_l2(const T* a, const T* b, int dim) {
	float distance = 0.0f;
	#pragma omp simd reduction(+:distance)
	for (int idx = 0; idx < dim; ++idx) {
		distance += (a[idx] - b[idx]) * (a[idx] - b[idx]);
	}
	return distance;
}

template <typename T>
inline void vector_sum(const T* a, const T* b, T* result, int dim) {
	#pragma omp simd
	for (int idx = 0; idx < dim; ++idx) {
		result[idx] = a[idx] + b[idx];
	}
}

void l2_norm_data(std::vector<float>& data, int dim);
std::vector<float> l2_norm_data(const std::vector<float>& data, int dim);
