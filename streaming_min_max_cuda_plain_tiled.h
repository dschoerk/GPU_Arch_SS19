#ifndef STREAMING_MIN_MAX_CUDA_PLAIN_TILED_H
#define STREAMING_MIN_MAX_CUDA_PLAIN_TILED_H

#include "streaming_min_max_algorithm_interface.h"

class streaming_min_max_cuda_plain_tiled:
  public streaming_min_max_algorithm_interface
{
public:

    std::string_view get_name(
	) const;

    bool check_against_reference(
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
