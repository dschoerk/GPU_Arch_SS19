#ifndef STREAMING_MIN_MAX_THRUST_NAIVE_CUDA_CUH
#define STREAMING_MIN_MAX_THRUST_NAIVE_CUDA_CUH

#include <vector>

extern void streaming_min_max_thrust_naive_calc(
    std::vector<float> const & array,
    unsigned int width,
    std::vector<float> & minvalues,
    std::vector<float> & maxvalues
);

#endif // STREAMING_MIN_MAX_THRUST_CUDA_CUH
