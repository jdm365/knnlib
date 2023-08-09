#include <immintrin.h>
#include <limits>

#include "../include/sort.h"

int argmax(float *a, int n) {
    float max_val = -float(std::numeric_limits<float>::infinity());
    int idx = 0;

    __m256 p = _mm256_set1_ps(max_val);

    int remainder = n % 32;
    int limit = n - remainder;

    for (int i = 0; i < limit; i += 32) {
        for (int k = 0; k < 32; k += 8) {
            __m256 y = _mm256_load_ps(&a[i + k]);
            __m256 mask = _mm256_cmp_ps(p, y, _CMP_LT_OS);
            if (!_mm256_testz_ps(mask, mask)) {
                idx = i + k;
                for (int j = i; j < i + 32; ++j) {
                    max_val = (a[j] > max_val ? a[j] : max_val);
                }
                p = _mm256_set1_ps(max_val);
            }
        }
    }

    // Handle the remaining elements
    for (int i = limit; i < n; ++i) {
        if (a[i] > max_val) {
            max_val = a[i];
            idx = i;
        }
    }

    return idx;
}
