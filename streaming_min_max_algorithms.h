#ifndef STREAMING_MIN_MAX_ALGORIHTMS_H
#define STREAMING_MIN_MAX_ALGORITHMS_H

#include "streaming_min_max_algorithm_interface.h"

#include <memory>

extern std::vector<std::unique_ptr<streaming_min_max_algorithm_interface> > algorithms;

#endif
