#include "streaming_min_max_thrust_cuda.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <algorithm>

// alias dptr = thrust::device_ptr<float>;

struct ShiftedWindowMinMaxStep
{
    int win_size;
    thrust::device_ptr<float> data;
    thrust::device_ptr<float> out_max;
    thrust::device_ptr<float> out_min;
    
    ShiftedWindowMinMaxStep(
        thrust::device_ptr<float> _data, 
        int _win_size, 
        thrust::device_ptr<float> _out_min, 
        thrust::device_ptr<float> _out_max
    ) : data(_data), win_size(_win_size), out_min(_out_min), out_max(_out_max) {}

    __device__ void operator() (const int &idx)
    {
        // we use the parameter as index from a counting iterator
        // we store the min/max index

        float min = data[idx];
        float max = min;

        //printf("data %d: %f\n", idx, min);

        for(int i = 1; i < win_size; ++i)
        {
            float e = data[idx + i];
            min = fminf(min, e);
            max = fmaxf(max, e);
            
            // __syncthreads();
        }

        out_min[idx] = min;
        out_max[idx] = max;

        //out_min[idx] = idx;
    }
};

void streaming_min_max_thrust_calc(
    std::vector<float> const & array,
    unsigned int width,
    std::vector<float> & minvalues,
    std::vector<float> & maxvalues
)
{
    //thrust::host_vector<float> h_vec(array);

    // transfer data to the device
    thrust::device_vector<float> d_vec(array); // = h_vec;
    thrust::device_vector<float> d_minima(d_vec.size() - width + 1);
    thrust::device_vector<float> d_maxima(d_vec.size() - width + 1);
    

    
    // https://stackoverflow.com/questions/21761412/thrust-reduction-result-on-device-memory
    // thrust reductions return to host memory
    // we need to use reduce by key to keep it on host memory
    // otherwise we copy every min/max value from device to host separately -> super slow


    // stupid and simple version
    /*
    int i = 0;
    for(auto it = d_vec.begin(); it + width - 1 != d_vec.end(); ++it)
    {
        auto result = thrust::minmax_element(d_vec.begin() + i, d_vec.begin() + i + width);
        
        minvalues[i] = *result.first;
        maxvalues[i] = *result.second;
        ++i;
    }*/

    thrust::counting_iterator<int> c_begin(0);
    thrust::counting_iterator<int> c_end(d_vec.size() - width + 1); // inclusive end?

    //for all shifts
    {
        thrust::for_each(c_begin, c_end,
            ShiftedWindowMinMaxStep(
                d_vec.data(), 
                width, 
                d_minima.data(), 
                d_maxima.data()
            ));
    }

    thrust::copy(d_minima.begin(), d_minima.end(), minvalues.begin()); // is this efficient?
    thrust::copy(d_maxima.begin(), d_maxima.end(), maxvalues.begin());
    
    /*thrust::host_vector<float> h_minima(minvalues.begin(), minvalues.end());
    thrust::host_vector<float> h_maxima(maxvalues.begin(), maxvalues.end());

    h_minima = d_minima;
    h_maxima = d_maxima;*/




    /*for(float f : thrust::host_vector<float>(d_vec))
        std::cout << f << ", ";
    std::cout << std::endl;

    minvalues = d_minima;
    for(float f : minvalues )
        std::cout << f << ", ";
    std::cout << std::endl;

    maxvalues = d_maxima;
    for(float f : maxvalues)
        std::cout << f << ", ";
    std::cout << std::endl;*/

}
