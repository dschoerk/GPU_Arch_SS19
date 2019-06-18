#include "streaming_min_max_cuda_plain_cuda.cuh"

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

struct ShiftedWindowMinMaxStep2
{
    int win_size;
    int data_size;
    thrust::device_ptr<float> data;
    thrust::device_ptr<float> out_max;
    thrust::device_ptr<float> out_min;
    thrust::device_ptr<float> tmp_max;
    thrust::device_ptr<float> tmp_min;

    int win_size_log2; // next smaller power of 2
    
    ShiftedWindowMinMaxStep2(
        thrust::device_ptr<float> _data, 
        int _win_size, 
        int _data_size,
        thrust::device_ptr<float> _tmp_min, 
        thrust::device_ptr<float> _tmp_max,
        thrust::device_ptr<float> _out_min, 
        thrust::device_ptr<float> _out_max
    ) : data(_data), win_size(_win_size), out_min(_out_min), out_max(_out_max), tmp_min(_tmp_min), tmp_max(_tmp_max) , data_size(_data_size)
    {
        for(win_size_log2 = 0; 1 << win_size_log2 <= win_size; ++win_size_log2); // find next smaller power of 2 to win_size
        win_size_log2--;

        printf("win_size_log2: %d\n", win_size_log2);
        printf("summation steps: %d\n", win_size - (1 << win_size_log2));
    }

    __device__ void operator() (const int &k)
    {
        //if (k > 250)
        /*{
            printf("v: %d %f\n", k, data[k]);
        }*/
        
        for(int d = 0; d < win_size_log2; ++d) // naive and incomplete prefix sum
        {
            if(k + (1 << d) < data_size)
            {
                tmp_min[k] = fminf(tmp_min[k], tmp_min[k + (1 << d)]);
                tmp_max[k] = fmaxf(tmp_max[k], tmp_max[k + (1 << d)]);
            }

            __syncthreads();
        }

        if(k < data_size - win_size + 1) 
        {
            out_min[k] = tmp_min[k];
            out_max[k] = tmp_max[k];

            __syncthreads();

            for(int i = 0; i < win_size - (1 << win_size_log2); ++i) // sum up the rest
            {
                out_min[k] = fminf(out_min[k], tmp_min[k+i+1]);
                out_max[k] = fmaxf(out_max[k], tmp_max[k+i+1]);

                __syncthreads();
            }
        }
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
    thrust::device_vector<float> d_minima(d_vec.size());
    thrust::device_vector<float> d_maxima(d_vec.size());

    thrust::device_vector<float> d_tmp_minima(array);
    thrust::device_vector<float> d_tmp_maxima(array);
    

    
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

    //for all shifts
    {
        thrust::for_each(c_begin, c_end,
            ShiftedWindowMinMaxStep2(
                d_vec.data(), 
                width, 
                array.size(),
                d_tmp_minima.data(), 
                d_tmp_maxima.data(),
                d_minima.data(), 
                d_maxima.data()
            ));
    }

    thrust::copy(d_minima.begin(), d_minima.end() - width + 1, minvalues.begin()); // is this efficient?
    thrust::copy(d_maxima.begin(), d_maxima.end() - width + 1, maxvalues.begin());
    
    /*thrust::host_vector<float> h_minima(minvalues.begin(), minvalues.end());
    thrust::host_vector<float> h_maxima(maxvalues.begin(), maxvalues.end());

    h_minima = d_minima;
    h_maxima = d_maxima;*/


    for(int i=505; i < 514; i++)
        std::cout << i << ": " << array[i] << ", ";

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