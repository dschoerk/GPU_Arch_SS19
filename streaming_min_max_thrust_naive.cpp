#include "streaming_min_max_thrust_naive.h"
#include "streaming_min_max_thrust_naive_cuda.cuh"
#include "utils.h"

std::string_view streaming_min_max_thrust_naive::get_name(
    ) const
{
    return "thrust_naive";
}
  
bool streaming_min_max_thrust_naive::check_against_reference(
    ) const
{
    return true;
}

void streaming_min_max_thrust_naive::calc(
    std::vector<float> const & array,
    unsigned int width
    )
{
    maxvalues.clear();
    maxvalues.resize(array.size() - width + 1);
    minvalues.clear();
    minvalues.resize(array.size() - width + 1);

    streaming_min_max_thrust_naive_calc(array, width, minvalues, maxvalues);
}

std::vector<float> const & streaming_min_max_thrust_naive::get_max_values(
    ) const
{
    return maxvalues;
}

std::vector<float> const & streaming_min_max_thrust_naive::get_min_values(
    ) const
{
    return minvalues;
}
