
#include "utils.h"
#include "streaming_min_max_cuda_plain_tiled_cuda.cuh"

// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>

#include <nvToolsExt.h>

#include <cooperative_groups.h>

#define BLOCK_SIZE 1024

namespace cg = cooperative_groups;

/**
 * CUDA kernel device code
 *
 * Computes a streaming minimum and maximum of the values of array \a
 * d_array within a window of size \a width and stores these these
 * minima and maxima in \a d_min and \a d_max respectively.
 */
__global__ void streaming_min_max_cuda_plain_tiled_calc(
    float const * d_array,
    float * d_min,
	float * d_max,
	const unsigned int array_elements,
    unsigned int min_max_elements,
	unsigned int win_size,
	unsigned int win_size_log2
    )  
{


	// tiling by blocks
	// unsigned int tile = blockIdx.x;
	// unsigned int tile_offset = blockDim.x * blockIdx.x;

	const int BLOCK_IDX = blockIdx.x; 
	const int OUT_BLOCK_SIZE = BLOCK_SIZE - win_size + 1; // the number of results a block can compute
	const int k = threadIdx.x;

	//if(tile_offset == 0 && k == 0)
	//	printf("OUT_BLOCK_SIZE: %d BLOCK_SIZE: %d win_size: %d 2^win_size_log2: %d\n", OUT_BLOCK_SIZE, BLOCK_SIZE, win_size, (1<<win_size_log2));

	
	//if(BLOCK_IDX > 0)
	//	return;

	if(k >= array_elements)
		return;

	__shared__ float s_min[BLOCK_SIZE];
	__shared__ float s_max[BLOCK_SIZE];


	if( k + BLOCK_IDX * OUT_BLOCK_SIZE < array_elements )
	{
		float read = d_array[k + BLOCK_IDX * OUT_BLOCK_SIZE];
		s_min[k] = read;
		s_max[k] = read;
	}

	__syncthreads();
	
	for(int d = 0; d < win_size_log2; ++d) // naive and incomplete prefix sum
	{
		// pack this into separate kernels
		float min1, min2, max1, max2;
		
		if(k + (1 << d) < BLOCK_SIZE)
		{
			min1 = s_min[k];
			min2 = s_min[k + (1 << d)];
			max1 = s_max[k];
			max2 = s_max[k + (1 << d)];
		}
			
		__syncthreads();
		
		if(k + (1 << d) < BLOCK_SIZE)
		{
			if(min2 < min1)
				s_min[k] = min2;

			if(max2 > max1)
				s_max[k] = max2;
		}

			//printf("writing %d %f %f\n", k, min1, min2);
			
		__syncthreads();
	}

	if(OUT_BLOCK_SIZE * BLOCK_IDX + k < array_elements - win_size + 1 && k < OUT_BLOCK_SIZE)
	//if( k < )
	{
		float min = s_min[k];
		float max = s_max[k];

		__syncthreads();

		for(int i = 0; i < win_size - (1 << win_size_log2); ++i) // sum up the rest
		{
			min = fminf(min, s_min[k+i+1]);
			max = fmaxf(max, s_max[k+i+1]);

			__syncthreads();
		}

		//printf("out: %d blk: %d k: %d min: %f max: %f\n", OUT_BLOCK_SIZE * BLOCK_IDX + k, BLOCK_IDX, k, min, max);

		d_min[OUT_BLOCK_SIZE * BLOCK_IDX + k] = min;
		d_max[OUT_BLOCK_SIZE * BLOCK_IDX + k] = max;
	}
	
}


float * d_mem_tiled(NULL);

static void streaming_min_max_cuda_plain_clean_up(
    )
{
    cudaError_t err(cudaSuccess);

    if (d_mem_tiled != NULL)
    {
	TRACE(
	    "Freeing allocated device memory at 0x%lx ...\n",
	    (unsigned long) d_mem_tiled
	    );

	err = cudaFree(d_mem_tiled);

	if (err != cudaSuccess)
	{
	    ERROR_EXIT(
		"Failed to free allocated device memory at 0x%lx - %s",
		(unsigned long) d_mem_tiled,
		cudaGetErrorString(err)
		);
	}
    }

    d_mem_tiled = NULL;
}

void streaming_min_max_cuda_plain_tiled_calc(
    float const * h_array,
    float * h_min,
    float * h_max,
    unsigned int array_elements,
    unsigned int min_max_elements,
    unsigned int width
    )
{
PUSH_RANGE("h2d", 1)
	
    unsigned int const min_max_size = min_max_elements * sizeof(float);
    unsigned int const array_size = array_elements * sizeof(float);
    unsigned int const total_mem_size(array_size + 2 * min_max_size);
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
	"\tCooperative Launch: %d\n"
	"\tCompute Capability: %d.%d\n"
	"\tCan map host memory: %s\n",
	dev_count,
	0,
	dev_prop.name,
	dev_prop.totalGlobalMem,
	dev_prop.sharedMemPerBlock,
	dev_prop.regsPerBlock,
	dev_prop.warpSize,
	dev_prop.maxThreadsPerBlock,
	dev_prop.cooperativeLaunch,
	dev_prop.major, dev_prop.minor,
	(dev_prop.canMapHostMemory == 0) ? "no": "yes"
	);   

	int win_size_log2;
    for(win_size_log2 = 0; 1 << win_size_log2 <= width; ++win_size_log2); // find next smaller power of 2 to win_size
	win_size_log2--;
	

	const int OUT_BLOCK_SIZE = BLOCK_SIZE - width + 1;
	int const threadsPerBlock = BLOCK_SIZE; //dev_prop.maxThreadsPerBlock);
	
	const int output_blocks = min_max_elements / OUT_BLOCK_SIZE + OUT_BLOCK_SIZE;

	int const blocksPerGrid = output_blocks;//(((output_blocks * threadsPerBlock) + threadsPerBlock - 1) / threadsPerBlock);
	
    //
    // allocate device memory
    //

    TRACE(
	"Allocating %u bytes of device memory ...\n",
	total_mem_size
	);

    err = cudaMalloc((void **) &d_mem_tiled, total_mem_size);

    if (err != cudaSuccess)
    {
		streaming_min_max_cuda_plain_clean_up();

        ERROR_EXIT(
	    "Failed to allocate %u bytes of memory on device - %s",
	    total_mem_size,
	    cudaGetErrorString(err)
	    );
    }

    TRACE(
	"Successfully allocated %u bytes of device memory at 0x%lx ...\n",
	total_mem_size,
	(unsigned long) d_mem_tiled
	);

    //
    // initialize pointers to subregions
    //

    float *d_array(d_mem_tiled);
    float *d_min(d_mem_tiled + array_elements);
    float *d_max(d_min + min_max_elements);

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
	    "Failed to copy %u bytes of input vector to device memory - %s",
	    array_size,
	    cudaGetErrorString(err)
	    );
    }

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

    streaming_min_max_cuda_plain_tiled_calc<<<blocksPerGrid, threadsPerBlock>>>(
	d_array,
	d_min,
	d_max,
	array_elements,
	min_max_elements,
	width,
	win_size_log2
	);   

cudaDeviceSynchronize();
POP_RANGE

PUSH_RANGE("d2h", 3)
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

	TRACE("check1\n");
    TRACE(
	"Copying %u bytes of output data from device memory 0x%lx into vector 0x%lx ...\n",
	min_max_size,
	(unsigned long) d_min,
	(unsigned long) h_min
	);

	TRACE("cudaMemcpy %d\n", min_max_size);
	err = cudaMemcpy(h_min, d_min, min_max_size, cudaMemcpyDeviceToHost);
	TRACE("check2\n");

    if (err != cudaSuccess)
    {
		streaming_min_max_cuda_plain_clean_up();
	
        ERROR_EXIT(
	    "Failed to copy %u bytes to input vector from device memory - %s",
	    min_max_size,
	    cudaGetErrorString(err)
	    );
	}
	
	TRACE("check3");

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
	    "Failed to copy %u bytes to input vector from device memory - %s",
	    min_max_size,
	    cudaGetErrorString(err)
	    );
    }

	streaming_min_max_cuda_plain_clean_up();
	
POP_RANGE
}
