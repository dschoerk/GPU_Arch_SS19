#include "streaming_min_max_thrust.h"

char const * streaming_min_max_thrust::get_name(
    ) const
{
    return "thrust";
}
  
void streaming_min_max_thrust::calc(
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

std::vector<float> const & streaming_min_max_thrust::get_max_values(
    ) const
{
    return maxvalues;
}

std::vector<float> const & streaming_min_max_thrust::get_min_values(
    ) const
{
    return minvalues;
}
