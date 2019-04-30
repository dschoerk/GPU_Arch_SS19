#include "streaming_min_max_algorithms.h"

#include "streaming_min_max_lemire.h"
#include "streaming_min_max_cuda_plain.h"
#include "streaming_min_max_thrust.h"
#include "streaming_min_max_cuda_streams.h"

std::unique_ptr<streaming_min_max_algorithm_interface> algorithms_array[]
{
    std::unique_ptr<streaming_min_max_lemire>(new streaming_min_max_lemire),
    std::unique_ptr<streaming_min_max_cuda_plain>(new streaming_min_max_cuda_plain),
    std::unique_ptr<streaming_min_max_thrust>(new streaming_min_max_thrust),
    std::unique_ptr<streaming_min_max_cuda_streams>(new streaming_min_max_cuda_streams)
};

const size_t algorithms_array_size = sizeof(algorithms_array) / sizeof(algorithms_array[0]);
