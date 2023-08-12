#pragma once

#include <immintrin.h>


int filter_avx2(float* a, float* b, float* idxs, float _cutoff, int N);
float dot_product_avx2(const float* a, const float* b, int n);
int dot_product_filter_avx2(float* a, float* b, float* idxs, float cutoff, int n);

struct Precalc {
    alignas(64) int permutation[256][8];

    constexpr Precalc(): permutation{} {
        for (int m = 0; m < 256; ++m) {
            int k = 0;
            for (int idx = 0; idx < 8; ++idx)
                if (m >> idx & 1)
                    permutation[m][k++] = idx;
        }
    }
};

constexpr Precalc T;
