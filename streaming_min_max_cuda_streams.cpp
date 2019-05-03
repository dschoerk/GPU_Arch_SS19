#include "streaming_min_max_cuda_streams.h"
#include "utils.h"

std::string_view streaming_min_max_cuda_streams::get_name(
    ) const
{
    return "cuda streams";
}
  
bool streaming_min_max_cuda_streams::check_against_reference(
    ) const
{
    return false;
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

    TRACE("TODO: Implement this one ...\n");
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
