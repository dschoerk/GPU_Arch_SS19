#ifndef STREAMING_MIN_MAX_ALGORIHTM_INTERFACE_H
#define STREAMING_MIN_MAX_ALGORIHTM_INTERFACE_H

#include <vector>

class streaming_min_max_algorithm_interface
{
public:

    virtual char const * get_name(
	) const = 0;
    
    virtual void calc(
	std::vector<float> const & array,
	unsigned int width
	) = 0;
    
    virtual std::vector<float> const & get_max_values(
	) const = 0;
    
    virtual std::vector<float> const & get_min_values(
	) const = 0;
    
    virtual ~streaming_min_max_algorithm_interface(
	) = default;
};

#endif
