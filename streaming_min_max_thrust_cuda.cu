#include "streaming_min_max_thrust_cuda.cuh"

#include "utils.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_malloc.h>
#include <algorithm>
#include <cooperative_groups.h>

#include <nvToolsExt.h>

#include <stdio.h>

namespace cg = cooperative_groups;

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

struct ShiftedWindowMinMaxStep2
{
    int win_size;
    int data_size;
    int d;
    thrust::device_ptr<float> out_max;
    thrust::device_ptr<float> out_min;
    thrust::device_ptr<float> in_max;
    thrust::device_ptr<float> in_min;

    int win_size_log2; // next smaller power of 2
    
    ShiftedWindowMinMaxStep2(
        int _win_size, 
        int _data_size,
        int _d,
        thrust::device_ptr<float> _in_min, 
        thrust::device_ptr<float> _in_max,
        thrust::device_ptr<float> _out_min, 
        thrust::device_ptr<float> _out_max
    ) : win_size(_win_size), out_min(_out_min), out_max(_out_max), in_min(_in_min), in_max(_in_max) , data_size(_data_size), d(_d)
    {
    }

    __device__ void operator() (const int &k)
    {
        
        //for(int d = 0; d < win_size_log2; ++d) // naive and incomplete prefix sum
        {
            // pack this into separate kernels
            if(k + (1 << d) < data_size)
            {
                out_min[k] = fminf(in_min[k], in_min[k + (1 << d)]);
                out_max[k] = fmaxf(in_max[k], in_max[k + (1 << d)]);
            }
        }
    }
};

struct SummationStep
{
    int win_size;
    int data_size;
    thrust::device_ptr<float> out_max;
    thrust::device_ptr<float> out_min;
    thrust::device_ptr<float> in_max;
    thrust::device_ptr<float> in_min;

    int win_size_log2; // next smaller power of 2
    
    SummationStep(
        int _win_size_log2, 
        int _win_size, 
        int _data_size,
        thrust::device_ptr<float> _in_min, 
        thrust::device_ptr<float> _in_max,
        thrust::device_ptr<float> _out_min, 
        thrust::device_ptr<float> _out_max
    ) : win_size_log2(_win_size_log2), win_size(_win_size), out_min(_out_min), out_max(_out_max), in_min(_in_min), in_max(_in_max) , data_size(_data_size)
    {
    }

    __device__ void operator() (const int &k)
    {
        if(k < data_size) 
        {
            float min = in_min[k];
            float max = in_max[k];

            //__syncthreads();

            for(int i = 0; i < win_size - (1 << win_size_log2); ++i) // sum up the rest
            {
                min = fminf(min, in_min[k+i+1]);
                max = fmaxf(max, in_max[k+i+1]);
            }

            out_min[k] = min;
            out_max[k] = max;
        }
    }
};

void streaming_min_max_thrust_calc(
    std::vector<float> const & array,
    unsigned int win_size,
    std::vector<float> & minvalues,
    std::vector<float> & maxvalues
)
{
    //thrust::host_vector<float> h_vec(array);

    // transfer data to the device
PUSH_RANGE("h2d", 1)
    thrust::device_vector<float> d_vec(array); // = h_vec;
    thrust::device_vector<float> d_minima(d_vec.size());
    thrust::device_vector<float> d_maxima(d_vec.size());

    thrust::device_vector<float> d_in_minima(d_vec.size(), 0); //(array);
    thrust::device_vector<float> d_in_maxima(d_vec.size(), 0); //(array);

    thrust::copy(d_vec.begin(), d_vec.end(), d_in_minima.begin());
    thrust::copy(d_vec.begin(), d_vec.end(), d_in_maxima.begin());
    

    
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
    //thrust::counting_iterator<int> c_end(d_vec.size() - width + 1); // inclusive end?
    thrust::counting_iterator<int> c_end(d_vec.size());

    int win_size_log2;
    for(win_size_log2 = 0; 1 << win_size_log2 <= win_size; ++win_size_log2); // find next smaller power of 2 to win_size
    win_size_log2--;

    //for all shifts

    thrust::device_vector<float>& input_min = d_in_minima;
    thrust::device_vector<float>& input_max = d_in_maxima;
    
    thrust::device_vector<float>& output_min = d_minima;
    thrust::device_vector<float>& output_max = d_maxima;

POP_RANGE

PUSH_RANGE("kernel", 2)
    for(int d = 0; d < win_size_log2; ++d)
    {
        thrust::for_each(c_begin, c_end,
            ShiftedWindowMinMaxStep2(
                win_size, 
                array.size(),
                d,
                input_min.data(), 
                input_max.data(),
                output_min.data(), 
                output_max.data()
            ));
        

        std::swap(input_min, output_min);
        std::swap(input_max, output_max);
    
        /*thrust::device_ptr<float> tmp = input_min;
        input_min = output_min;
        output_min = tmp;*/
    }
POP_RANGE

    // output is in input when finished
PUSH_RANGE("kernel", 4)
    thrust::for_each(c_begin, c_end,
        SummationStep(
            win_size_log2, 
            win_size,
            array.size(),
            input_min.data(), 
            input_max.data(),
            output_min.data(), 
            output_max.data()
        ));
cudaDeviceSynchronize();
POP_RANGE



    // output in output after summation step

PUSH_RANGE("d2h", 3)
    thrust::copy(output_min.begin(), output_min.end() - win_size + 1, minvalues.begin()); // is this efficient?
    thrust::copy(output_max.begin(), output_max.end() - win_size + 1, maxvalues.begin());
POP_RANGE

    /*thrust::host_vector<float> h_minima(minvalues.begin(), minvalues.end());
    thrust::host_vector<float> h_maxima(maxvalues.begin(), maxvalues.end());

    h_minima = d_minima;
    h_maxima = d_maxima;*/


    /*for(float f : thrust::host_vector<float>(d_vec))
        std::cout << f << ", ";
    std::cout << std::endl << std::endl;

    for(float f : minvalues )
        std::cout << f << ", ";
    std::cout << std::endl << std::endl;

    for(float f : maxvalues)
        std::cout << f << ", ";
    std::cout << std::endl << std::endl;*/

}
