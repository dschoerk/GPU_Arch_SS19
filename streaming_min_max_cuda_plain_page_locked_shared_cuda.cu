#include "utils.h"
#include "streaming_min_max_cuda_plain_page_locked_shared_cuda.cuh"

// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>

//
// Obtained blocksize from device query - needs to be adapted to device
//
#define BLOCK_SIZE 1024

//
// Setting maximum value for window with to blocksize
//
#define MAX_WIDTH BLOCK_SIZE

/**
 * CUDA kernel device code
 *
 * Computes a streaming minimum and maximum of the values of array \a
 * d_array within a window of size \a width and stores these these
 * minima and maxima in \a d_min and \a d_max respectively.
 */
__global__ void streaming_min_max_cuda_plain_page_locked_shared(
    float const * d_array,
    float * d_min,
    float * d_max,
    unsigned int min_max_elements,  
    unsigned int width
    )  
{
    int const thread_index(blockDim.x * blockIdx.x + threadIdx.x);
    float min, max;

    //
    // use shared memory as cache for the relevant part of d_array
    //
    __shared__ float d_array_cache[BLOCK_SIZE + MAX_WIDTH];

    const int max_thread_index = min_max_elements + width;
	
    if (thread_index < max_thread_index)
    {
		d_array_cache[threadIdx.x] = d_array[thread_index];

		if ((threadIdx.x < width) && (thread_index + BLOCK_SIZE < max_thread_index))
		{
			d_array_cache[BLOCK_SIZE + threadIdx.x] = d_array[BLOCK_SIZE + thread_index];
		}
    }

    //
    // synchronized threads to ensure that cached data is available
    //
    __syncthreads();
    
    //
    // now work on cache
    //
    if (thread_index < min_max_elements)
    {
		min = d_array_cache[threadIdx.x];
		max = d_array_cache[threadIdx.x];
		
		for (int i = 1; i < width; ++i)
		{
			float current = d_array_cache[threadIdx.x + i];

			//
			// Tried the following here, but that deteriorated the performance
			//
			// min = fminf(current, min);
			//
			if (current < min)
			{
			min = current;
			}

			//
			// Tried the following here, but that deteriorated the performance
			//
			// max = fmaxf(current, max);
			//
			if (current > max)
			{
			max = current;
			}
		}

		d_min[thread_index] = min;
		d_max[thread_index] = max;
    }
}


static float *d_array(nullptr);
static float *d_min(nullptr);
static float *d_max(nullptr);

static float const *gh_array(nullptr);
static float *gh_min(nullptr);
static float *gh_max(nullptr);

static void streaming_min_max_cuda_plain_clean_up(
    )
{
    cudaError_t err(cudaSuccess);

    if (d_array != nullptr)
    {
		TRACE(
			"Unregistering host memory at 0x%lx ...\n",
			(unsigned long) gh_array
			);

		err = cudaHostUnregister((void *) gh_array);

		if (err != cudaSuccess)
		{
			ERROR_EXIT(
			"Failed to unregister memory at 0x%lx - %s",
			(unsigned long) gh_array,
			cudaGetErrorString(err)
			);
		}
    }

    d_array = nullptr;

    if (d_min != nullptr)
    {
		TRACE(
			"Unregistering host memory at 0x%lx ...\n",
			(unsigned long) gh_min
			);

		err = cudaHostUnregister(gh_min);

		if (err != cudaSuccess)
		{
			ERROR_EXIT(
			"Failed to unregister memory at 0x%lx - %s",
			(unsigned long) gh_min,
			cudaGetErrorString(err)
			);
		}
    }

    d_min = nullptr;

    if (d_max != nullptr)
    {
		TRACE(
			"Unregistering host memory at 0x%lx ...\n",
			(unsigned long) gh_max
			);

		err = cudaHostUnregister(gh_max);

		if (err != cudaSuccess)
		{
			ERROR_EXIT(
			"Failed to unregister memory at 0x%lx - %s",
			(unsigned long) gh_max,
			cudaGetErrorString(err)
			);
		}
    }

    d_max = nullptr;
}

static void register_host_memory(
    float const * h_mem,
    float * &d_mem,
    unsigned int size
    )
{
    cudaError_t err(cudaSuccess);

    TRACE(
	"Registering %u bytes of host memory at 0x%lx for use by CUDA ...\n",
	size,
	(unsigned long) h_mem
	);

    err = cudaHostRegister((void *) h_mem, size, cudaHostRegisterMapped);
    
    if (err != cudaSuccess)
    {
		streaming_min_max_cuda_plain_clean_up();

        ERROR_EXIT(
	    "Failed to register %u bytes of host memory at 0x%lx for use with CUDA - %s",
	    size,
	    (unsigned long) h_mem,
	    cudaGetErrorString(err)
	    );
    }

    TRACE(
	"Successfully registered %u bytes of host memory at 0x%lx for use with CUDA ...\n",
	size,
	(unsigned long) h_mem
	);

    TRACE(
	"Obtaining device pointer for %u bytes of host memory at 0x%lx  ...\n",
	size,
	(unsigned long) h_mem
	);

    err = cudaHostGetDevicePointer(&d_mem, (void *) h_mem, 0);

    if (err != cudaSuccess)
    {
		streaming_min_max_cuda_plain_clean_up();

        ERROR_EXIT(
	    "Failed to obtain device pointer for %u bytes of host memory at 0x%lx - %s",
	    size,
  	    (unsigned long) h_mem,
	    cudaGetErrorString(err)
	    );
    }

    TRACE(
	"Successfully obtained device pointer 0x%lx for %u bytes of host memory at 0x%lx ...\n",
	(unsigned long) d_mem,
	size,
	 (unsigned long) h_mem
	);
}

void streaming_min_max_cuda_plain_page_locked_shared_calc(
    float const * h_array,
    float * h_min,
    float * h_max,
    unsigned int array_elements,
    unsigned int min_max_elements,
    unsigned int width
    )
{
PUSH_RANGE("h2d", 1)
    unsigned int const min_max_size(min_max_elements * sizeof(float));
    unsigned int const array_size(array_elements * sizeof(float));
    cudaError_t err(cudaSuccess);
    int dev_count(0);
    cudaDeviceProp dev_prop;

    //
    // query device properties
    //

    (void) cudaGetDeviceCount(&dev_count);
    (void) cudaGetDeviceProperties(&dev_prop, 0);

    TRACE(
	"Found %d devices and queried the following properties for device %d ...\n"
	"\tName: %s\n"
	"\tGlobal memory [bytes]: %u\n"
	"\tShared memory per block [bytes]: %u\n"
	"\tRegisters per block: %u\n"
	"\tWarp size: %u\n"
	"\tMaximum threads per block: %u\n"
	"\tCan map host memory: %s\n",
	dev_count,
	0,
	dev_prop.name,
	dev_prop.totalGlobalMem,
	dev_prop.sharedMemPerBlock,
	dev_prop.regsPerBlock,
	dev_prop.warpSize,
	dev_prop.maxThreadsPerBlock,
	(dev_prop.canMapHostMemory == 0) ? "no": "yes"
	);   
    
    int const threadsPerBlock(dev_prop.maxThreadsPerBlock);
    int const blocksPerGrid((array_size + threadsPerBlock - 1) / threadsPerBlock);

    //
    // sanity check block size and window width against statically configured
    // maximum values
    //
    if (threadsPerBlock > BLOCK_SIZE)
    {
	std::cout << "Number of threads per block " << threadsPerBlock
		  << " exceeds statically configured maximum block size "
		  << BLOCK_SIZE << '\n'
		  << "Skipping execution! - Increase BLOCK_SIZE and re-build!\n";

	return;
    }

    if (width > MAX_WIDTH)
    {
	std::cout << "Window width " << width
		  << " exceeds statically configured maximum window width "
		  << MAX_WIDTH << '\n'
		  << "Skipping execution! - Increase MAX_WIDTH and re-build!\n";
	
	return;
    }

    //
    // register host memory with device
    //

    register_host_memory(
	h_array,
	d_array,
	array_size
	);
    gh_array = h_array;
    
    register_host_memory(
	h_min,
	d_min,
	min_max_size
	);
    gh_min = h_min;

    register_host_memory(
	h_max,
	d_max,
	min_max_size
	);
    gh_max = h_max;

    //
    // launch the CUDA kernel
    //

    TRACE(
	"Launching CUDA kernel with %d blocks of %d threads ...\n",
	blocksPerGrid,
	threadsPerBlock
	);   

POP_RANGE

PUSH_RANGE("kernel", 2)
    streaming_min_max_cuda_plain_page_locked_shared<<<blocksPerGrid, threadsPerBlock>>>(
	d_array,
	d_min,
	d_max,
	min_max_elements,
	width
	);    
	err = cudaGetLastError();
	
cudaDeviceSynchronize();
POP_RANGE

PUSH_RANGE("d2h", 3)

    if (err != cudaSuccess)
    {
	streaming_min_max_cuda_plain_clean_up();
	
        ERROR_EXIT(
	    "Failed to launch kernel - %s",
	    cudaGetErrorString(err)
	    );
    }

    TRACE(
	"Waiting for CUDA kernel to finish ...\n"
	);   

    err = cudaDeviceSynchronize();
    
    if (err != cudaSuccess)
    {
	streaming_min_max_cuda_plain_clean_up();
	
        ERROR_EXIT(
	    "Failed to wait for CUDA kernel to finish - %s",
	    cudaGetErrorString(err)
	    );
    }

	streaming_min_max_cuda_plain_clean_up();
	
POP_RANGE
}
