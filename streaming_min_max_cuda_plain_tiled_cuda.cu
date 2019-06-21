#include "utils.h"
#include "streaming_min_max_cuda_plain_tiled_cuda.cuh"

// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>

#include <nvToolsExt.h>


#define BLOCK_SIZE 1024


static float *d_array_tiled(nullptr);
static float *d_min_tiled(nullptr);
static float *d_max_tiled(nullptr);

static float const *gh_array_tiled(nullptr);
static float *gh_min_tiled(nullptr);
static float *gh_max_tiled(nullptr);

static void streaming_min_max_cuda_plain_clean_up(
)
{
	cudaError_t err(cudaSuccess);

	if (d_array_tiled != nullptr)
	{
		TRACE(
			"Unregistering host memory at 0x%lx ...\n",
			(unsigned long) gh_array_tiled
			);

		err = cudaHostUnregister((void *) gh_array_tiled);

		if (err != cudaSuccess)
		{
			ERROR_EXIT(
			"Failed to unregister memory at 0x%lx - %s",
			(unsigned long) gh_array_tiled,
			cudaGetErrorString(err)
			);
		}
	}

	d_array_tiled = nullptr;

	if (d_min_tiled != nullptr)
	{
		TRACE(
			"Unregistering host memory at 0x%lx ...\n",
			(unsigned long) gh_min_tiled
			);

		err = cudaHostUnregister(gh_min_tiled);

		if (err != cudaSuccess)
		{
			ERROR_EXIT(
			"Failed to unregister memory at 0x%lx - %s",
			(unsigned long) gh_min_tiled,
			cudaGetErrorString(err)
			);
		}
	}

	d_min_tiled = nullptr;

	if (d_max_tiled != nullptr)
	{
		TRACE(
			"Unregistering host memory at 0x%lx ...\n",
			(unsigned long) gh_max_tiled
			);

		err = cudaHostUnregister(gh_max_tiled);

		if (err != cudaSuccess)
		{
			ERROR_EXIT(
			"Failed to unregister memory at 0x%lx - %s",
			(unsigned long) gh_max_tiled,
			cudaGetErrorString(err)
			);
		}
	}

	d_max_tiled = nullptr;
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

/**
 * CUDA kernel device code
 *
 * Computes a streaming minimum and maximum of the values of array \a
 * d_array_tiled within a window of size \a width and stores these these
 * minima and maxima in \a d_min_tiled and \a d_max_tile respectively.
 */
__global__ void streaming_min_max_cuda_plain_tiled_calc(
    float const * d_array_tiled,
    float * d_min_tiled,
	float * d_max_tiled,
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
		float read = d_array_tiled[k + BLOCK_IDX * OUT_BLOCK_SIZE];
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

		d_min_tiled[OUT_BLOCK_SIZE * BLOCK_IDX + k] = min;
		d_max_tiled[OUT_BLOCK_SIZE * BLOCK_IDX + k] = max;
	}
	
}


// float * d_mem_tiled(NULL);




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

    register_host_memory(
	h_array,
	d_array_tiled,
	array_size
	);
	gh_array_tiled = h_array;
	
	register_host_memory(
	h_min,
	d_min_tiled,
	min_max_size
	);
	gh_min_tiled = h_min;

	register_host_memory(
	h_max,
	d_max_tiled,
	min_max_size
	);
	gh_max_tiled = h_max;

    //
    // initialize pointers to subregions
    //

    /*float *d_array_tiled(d_mem_tiled);
    float *d_min_tiled(d_mem_tiled + array_elements);
	float *d_max_tile(d_min_tiled + min_max_elements);*/

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
	d_array_tiled,
	d_min_tiled,
	d_max_tiled,
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

    

	streaming_min_max_cuda_plain_clean_up();
	
POP_RANGE
}
