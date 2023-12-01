#pragma once

#include <vector>

struct ProductQuantizer {
	alignas(16) std::vector<float> centroids;
	int num_subvectors;
	int num_centroids;
	int dim;
	bool is_trained = false;
};

ProductQuantizer train_quantizer(
		const std::vector<float>& data,
		int dim,
		int num_subvectors = 4,
		int num_centroids = 32
		);
void train_quantizer(
		ProductQuantizer& quantizer,
		const std::vector<float>& data,
		const int dim,
		const int num_subvectors = 4,
		const int num_centroids = 32
		);

std::vector<uint8_t> product_quantize(
		const std::vector<float>& data,
		const ProductQuantizer& quantizer
		);
