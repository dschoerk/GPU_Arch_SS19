#include "streaming_min_max_thrust.h"
#include "streaming_min_max_thrust_cuda.cuh"
#include "utils.h"

std::string_view streaming_min_max_thrust::get_name(
    ) const
{
    return "thrust";
}
  
bool streaming_min_max_thrust::check_against_reference(
    ) const
{
    return false;
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

    streaming_min_max_thrust_calc(array, width, minvalues, maxvalues);
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
