#include "streaming_min_max_cuda_plain_cuda.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <algorithm>



void streaming_min_max_thrust_calc(
    std::vector<float> const & array,
    unsigned int width,
    std::vector<float> & minvalues,
    std::vector<float> & maxvalues
)
{
    thrust::host_vector<float> h_vec(array);

    // transfer data to the device
    thrust::device_vector<float> d_vec = h_vec;

    // https://stackoverflow.com/questions/21761412/thrust-reduction-result-on-device-memory
    // thrust reductions return to host memory
    // we need to use reduce by key to keep it on host memory
    // otherwise we copy every min/max value from device to host separately -> super slow


    int i = 0;
    for(auto it = d_vec.begin(); it + width - 1 != d_vec.end(); ++it)
    {
        auto result = thrust::minmax_element(d_vec.begin() + i, d_vec.begin() + i + width);
        //float min = *test.first;

        minvalues[i] = *result.first;
        maxvalues[i] = *result.second;
        ++i;

        //printf("%d\n", i);
    }

    // transfer data back to host
    //thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}