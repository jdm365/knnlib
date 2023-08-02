#pragma once

#include <vector>
#include <cblas.h>

template <typename T>
float distance_l2(const T* a, const T* b, int dim) {
	float distance = 0.0f;
	for (int idx = 0; idx < dim; ++idx) {
		distance += (a[idx] - b[idx]) * (a[idx] - b[idx]);
	}
	return distance;
}

template <typename T>
void vector_sum(const T* a, const T* b, T* result, int dim) {
	for (int idx = 0; idx < dim; ++idx) {
		result[idx] = a[idx] + b[idx];
	}
}

void l2_norm_data(std::vector<float>& data, int dim);
