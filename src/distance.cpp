#include <cstring>
#include <vector>
#include <omp.h>
#include <cblas.h>

#include "../include/distance.h"

void l2_norm_data(std::vector<float>& data, int dim) {
	const float ETA = 1e-8;

	#pragma omp parallel for
	for (int idx = 0; idx < (int)data.size() / dim; ++idx) {
		float norm = cblas_snrm2(
				dim,
				data.data() + idx * dim,
				1
				);
		for (int d = 0; d < dim; ++d) {
			data[idx * dim + d] /= (norm + ETA);
		}
	}
}

std::vector<float> l2_norm_data(const std::vector<float>& data, int dim) {
	const float ETA = 1e-8;
	std::vector<float> normed_data(data.size());
	memcpy(normed_data.data(), data.data(), data.size() * sizeof(float));

	#pragma omp parallel for
	for (int idx = 0; idx < (int)data.size() / dim; ++idx) {
		float norm = cblas_snrm2(
				dim,
				data.data() + idx * dim,
				1
				);
		// Element-wise multiplication blas
		cblas_sscal(
				dim,
				1.0f / (norm + ETA),
				normed_data.data() + idx * dim,
				1
				);
	}
	return normed_data;
}
