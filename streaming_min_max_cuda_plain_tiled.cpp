#include "streaming_min_max_cuda_plain_tiled.h"
#include "streaming_min_max_cuda_plain_tiled_cuda.cuh"

std::string_view streaming_min_max_cuda_plain_tiled::get_name(
    ) const
{
    return "cuda plain - cuda tiled";
}
  
bool streaming_min_max_cuda_plain_tiled::check_against_reference(
    ) const
{
    return true;
}

void streaming_min_max_cuda_plain_tiled::calc(
    std::vector<float> const & array,
    unsigned int width
    )
{
    unsigned int const array_size(array.size());
    unsigned int const min_max_size(array_size - width + 1);
    
    maxvalues.clear();
    maxvalues.resize(min_max_size);
    minvalues.clear();
    minvalues.resize(min_max_size);

    streaming_min_max_cuda_plain_tiled_calc(
	array.data(),
	minvalues.data(),
	maxvalues.data(),
	array_size,
	min_max_size,
	width
    );
}    

std::vector<float> const & streaming_min_max_cuda_plain_tiled::get_max_values(
    ) const
{
    return maxvalues;
}

std::vector<float> const & streaming_min_max_cuda_plain_tiled::get_min_values(
    ) const
{
    return minvalues;
}
