#include "streaming_min_max_algorithms.h"

#include "streaming_min_max_lemire.h"
#include "streaming_min_max_cuda_plain_malloc.h"
#include "streaming_min_max_cuda_plain_page_locked.h"
#include "streaming_min_max_cuda_plain_page_locked_shared.h"
#include "streaming_min_max_thrust.h"
#include "streaming_min_max_cuda_streams.h"

// ugly hack workaround because C++17 does not allow for brace initialization
// of containers containing non-copyable types - pfffff
static std::unique_ptr<streaming_min_max_algorithm_interface> algorithms_array[]
{
    std::make_unique<streaming_min_max_lemire>(),
    std::make_unique<streaming_min_max_cuda_plain_malloc>(),
    std::make_unique<streaming_min_max_cuda_plain_page_locked>(),
    std::make_unique<streaming_min_max_cuda_plain_page_locked_shared>(),
    std::make_unique<streaming_min_max_thrust>(),
    std::make_unique<streaming_min_max_cuda_streams>()
};

std::vector<std::unique_ptr<streaming_min_max_algorithm_interface> > algorithms
{
    std::make_move_iterator(std::begin(algorithms_array)),
    std::make_move_iterator(std::end(algorithms_array))
};
