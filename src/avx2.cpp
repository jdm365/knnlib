#include <immintrin.h>

#include "../include/avx2.h"


int filter_avx2(float* a, float* b, float* idxs, float _cutoff, int N) {
    const __m256 cutoff = _mm256_set1_ps(_cutoff);
    int k = 0;

    for (int idx = 0; idx < N; idx += 8) {
        __m256 x = _mm256_loadu_ps(&a[idx]);

        __m256 mask_values = _mm256_cmp_ps(cutoff, x, _CMP_GT_OS);
        int mask = _mm256_movemask_ps(mask_values);

        __m256i permutation = _mm256_load_si256((const __m256i*) &T.permutation[mask]);

        x = _mm256_permutevar8x32_ps(x, permutation);
        _mm256_storeu_ps(&b[k], x);

        // Handling indices 
        __m256i index_values = _mm256_set_epi32(
				idx + 7,
				idx + 6,
				idx + 5,
				idx + 4,
				idx + 3,
				idx + 2,
				idx + 1,
				idx
			);
        __m256i permuted_idxs = _mm256_permutevar8x32_epi32(index_values, permutation);
        _mm256_storeu_si256((__m256i*)&idxs[k], permuted_idxs);

        k += __builtin_popcount(mask);
    }

    return k;
}

float dot_product_avx2(const float* a, const float* b, int dim) {
	__m256 sum = _mm256_setzero_ps();
	for (int idx = 0; idx < dim; idx += 8) {
		__m256 a_vec = _mm256_loadu_ps(&a[idx]);
		__m256 b_vec = _mm256_loadu_ps(&b[idx]);
		__m256 prod = _mm256_mul_ps(a_vec, b_vec);
		sum = _mm256_add_ps(sum, prod);
	}
	float final_sum = 0;
	for (int idx = 0; idx < 8; ++idx) {
		final_sum += sum[idx];
	}
	return final_sum;
}

/*
int dot_product_filter_avx2(
		float* a, 
		float* b, 
		float* distances,
		float* idxs, 
		float cutoff, 
		int n
		) {
	// Calc dot product
	// __m256 sum = dot_product_avx2(a, b, n);

	// Filter
	// int k = filter_avx2(a, distances, idxs, cutoff, n);
	// TODO: Fix and finish.
	return 0;
}
*/
