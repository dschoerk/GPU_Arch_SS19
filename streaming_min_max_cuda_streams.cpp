#include "streaming_min_max_cuda_streams.h"

char const * streaming_min_max_cuda_streams::get_name(
    ) const
{
    return "cuda streams";
}
  
void streaming_min_max_cuda_streams::calc(
    std::vector<float> const & array,
    unsigned int width
    )
{
    maxvalues.clear();
    maxvalues.resize(array.size() - width + 1);
    minvalues.clear();
    minvalues.resize(array.size() - width + 1);

    // TODO: Insert code here ...
}

std::vector<float> const & streaming_min_max_cuda_streams::get_max_values(
    ) const
{
    return maxvalues;
}

std::vector<float> const & streaming_min_max_cuda_streams::get_min_values(
    ) const
{
    return minvalues;
}
