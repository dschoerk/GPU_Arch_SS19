#include "streaming_min_max_cuda_plain_malloc.h"
#include "streaming_min_max_cuda_plain_malloc_cuda.cuh"

std::string_view streaming_min_max_cuda_plain_malloc::get_name(
    ) const
{
    return "cuda plain - cuda malloc";
}
  
bool streaming_min_max_cuda_plain_malloc::check_against_reference(
    ) const
{
    return true;
}

void streaming_min_max_cuda_plain_malloc::calc(
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

    streaming_min_max_cuda_plain_malloc_calc(
	array.data(),
	minvalues.data(),
	maxvalues.data(),
	array_size,
	min_max_size,
	width
    );
}    

std::vector<float> const & streaming_min_max_cuda_plain_malloc::get_max_values(
    ) const
{
    return maxvalues;
}

std::vector<float> const & streaming_min_max_cuda_plain_malloc::get_min_values(
    ) const
{
    return minvalues;
}
