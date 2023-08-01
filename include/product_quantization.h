#pragma once

#include <vector>

std::vector<uint8_t> product_quantize(
		const std::vector<float>& data,
		int dim,
		int num_subvectors = 32,
		int num_centroids = 32
		);

