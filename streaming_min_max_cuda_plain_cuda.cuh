#ifndef STREAMING_MIN_MAX_CUDA_PLAIN_CUDA_CUH
#define STREAMING_MIN_MAX_CUDA_PLAIN_CUDA_CUH

extern void streaming_min_max_cuda_plain_calc(
    float const * h_array,
    float * h_min,
    float * h_max,
    unsigned int array_elements,
    unsigned int min_max_elements,
    unsigned int width
    );

#endif
