#include <vector>
#include <omp.h>
#include <cblas.h>

#include "../include/distance.h"
#include "../include/exact.h"


void FlatIndexL2::add(std::vector<float>& data) {
	this->data.insert(this->data.end(), data.begin(), data.end());

	l2_norm_data(this->data, this->dim);
}


std::vector<std::vector<std::pair<float, int>>> FlatIndexL2::search(const std::vector<float>& query, int k) {
	return get_exact_knn_blas(query, this->data, this->dim, k);
}

