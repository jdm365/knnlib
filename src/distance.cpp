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
		_Pragma("clang loop vectorize(enable) interleave(enable)")
		for (int d = 0; d < dim; ++d) {
			data[idx * dim + d] /= (norm + ETA);
		}
	}
}
