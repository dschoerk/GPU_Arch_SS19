#include "utils.h"
#include "streaming_min_max_cuda_plain_cuda.cuh"

// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime.h>
#include <helper_cuda.h>

/**
 * CUDA kernel device code
 *
 * Computes a streaming minimum and maximum of the values of array \a
 * d_array within a window of size \a width and stores these these
 * minima and maxima in \a d_min and \a d_max respectively.
 */
__global__ void streamingMinMax(
    float const * d_array,
    float * d_min,
    float * d_max,
    unsigned int array_size,
    unsigned int min_max_size,
    unsigned int width
    )  
{
    int const thread_index(blockDim.x * blockIdx.x + threadIdx.x);
    float min, max;

    if (thread_index < min_max_size)
    {
	min = d_array[thread_index];
	max = d_array[thread_index];
	
	for (int i = 1; i < width; ++i)
	{
	    float current = d_array[thread_index + i];
	    
	    if (current < min)
	    {
		min = current;
	    }

	    if (current > max)
	    {
		max = current;
	    }
	}

	d_min[thread_index] = min;
	d_max[thread_index] = max;
    }
}


float * d_mem(NULL);

static void streaming_min_max_cuda_plain_clean_up(
    )
{
    cudaError_t err(cudaSuccess);

    if (d_mem != NULL)
    {
	TRACE(
	    "Freeing allocated device memory at 0x%lx ...\n",
	    (unsigned long) d_mem
	    );

	err = cudaFree(d_mem);

	if (err != cudaSuccess)
	{
	    ERROR_EXIT(
		"Failed to free allocated device memory at 0x%lx - %s",
		(unsigned long) d_mem,
		cudaGetErrorString(err)
		);
	}
    }

    d_mem = NULL;
}

void streaming_min_max_cuda_plain_calc(
    float const * h_array,
    float * h_min,
    float * h_max,
    unsigned int array_size,
    unsigned int min_max_size,
    unsigned int width
    )
{
    unsigned int const total_mem_size((array_size + 2 * min_max_size) * sizeof(float));
    cudaError_t err(cudaSuccess);
    
    //
    // allocate device memory
    //

    TRACE(
	"Allocating %u bytes of device memory ...\n",
	total_mem_size
	);

    err = cudaMalloc((void **) &d_mem, total_mem_size * sizeof(float));

    if (err != cudaSuccess)
    {
	streaming_min_max_cuda_plain_clean_up();

        ERROR_EXIT(
	    "Failed to allocate %d bytes of memory on device - %s",
	    total_mem_size,
	    cudaGetErrorString(err)
	    );
    }

    TRACE(
	"Successfully allocated %u bytes of device memory at 0x%lx ...\n",
	total_mem_size,
	(unsigned long) d_mem
	);

    //
    // initialize pointers to subregions
    //

    float *d_array(d_mem);
    float *d_min(d_mem + array_size * sizeof(float));
    float *d_max(d_min + min_max_size * sizeof(float));

    //
    // copy input vector's data to device memory
    //

    TRACE(
	"Copying %u bytes of input data from vector 0x%lx into device memory 0x%lx ...\n",
	array_size,
	(unsigned long) h_array,
	(unsigned long) d_array
	);

    err = cudaMemcpy(d_array, h_array, array_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
	streaming_min_max_cuda_plain_clean_up();
	
        ERROR_EXIT(
	    "Failed to copy %d bytes of input vector to device memory - %s",
	    array_size,
	    cudaGetErrorString(err)
	    );
    }

    //
    // launch the CUDA kernel
    //

    int const threadsPerBlock(256);
    int const blocksPerGrid((array_size + threadsPerBlock - 1) / threadsPerBlock);

    TRACE(
	"Launching CUDA kernel with %d blocks of %d threads ...\n",
	blocksPerGrid,
	threadsPerBlock
	);   

    streamingMinMax<<<blocksPerGrid, threadsPerBlock>>>(
	d_array,
	d_min,
	d_max,
	array_size,
	min_max_size,
	width
	);    
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
	streaming_min_max_cuda_plain_clean_up();
	
        ERROR_EXIT(
	    "Failed to launch kernel - %s",
	    cudaGetErrorString(err)
	    );
    }

    //
    // copy output data from device memory into vectors
    //

    TRACE(
	"Copying %u bytes of output data from device memory 0x%lx into vector 0x%lx ...\n",
	min_max_size,
	(unsigned long) d_min,
	(unsigned long) h_min
	);

    err = cudaMemcpy(h_min, d_min, min_max_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
	streaming_min_max_cuda_plain_clean_up();
	
        ERROR_EXIT(
	    "Failed to copy %d bytes of input vector to device memory - %s",
	    min_max_size,
	    cudaGetErrorString(err)
	    );
    }

    TRACE(
	"Copying %u bytes of output data from device memory 0x%lx into vector 0x%lx ...\n",
	min_max_size,
	(unsigned long) d_max,
	(unsigned long) h_max
	);

    err = cudaMemcpy(h_max, d_max, min_max_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
	streaming_min_max_cuda_plain_clean_up();
	
        ERROR_EXIT(
	    "Failed to copy %d bytes of input vector to device memory - %s",
	    min_max_size,
	    cudaGetErrorString(err)
	    );
    }

    streaming_min_max_cuda_plain_clean_up();
}
