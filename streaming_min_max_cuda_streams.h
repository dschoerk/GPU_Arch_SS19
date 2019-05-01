#ifndef STREAMING_MIN_MAX_CUDA_STREAMS_H
#define STREAMING_MIN_MAX_CUDA_STREAMS_H

#include "streaming_min_max_algorithm_interface.h"

class streaming_min_max_cuda_streams:
  public streaming_min_max_algorithm_interface
{
public:

    std::string_view get_name(
	) const;

    void calc(
	std::vector<float> const & array,
	unsigned int width
	);

    std::vector<float> const & get_max_values(
	) const;

    std::vector<float> const & get_min_values(
	) const;

private:

    std::vector<float> maxvalues;
    std::vector<float> minvalues;
};

#endif
