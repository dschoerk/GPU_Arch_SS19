# GPU_Arch_SS19

Parallel implementations of a streaming minimum-maximum filter for
large datasets. Based on the algorithm proposed by Daniel Lemire in
the paper [Streaming maximum-minimum filter using nomore than three
comparisons per element](https://arxiv.org/pdf/cs/0610046.pdf).

## Sources

Currently the project consist of the following sources:

### Streaming min-max algorithm interface

`streaming_min_max_algorithm_interface.h` defines the abstract class
`streaming_min_max_algorithm_interface` which defines the interface
which needs to be implemented to have a [concrete
implementation](#Concrete-streaming-min-max-algorithms) of the
*streaming min-max algorithm* properly integrate with the
[benchmarking framework](#Streaming-min-max-benchmark-framework)
(`streaming_min_max_comparison`) and the [collection of
algorithms](#Streaming-min-max-algorithm-collection)
(`streaming_min_max_algorithms`). The interface hereby consists of the
following (pure virtual) member functions:

#### `get_name()`

This function is intended to return the name of the streaming min/max
algorithm implementation. This name is used in the output of the
benchmarking framework.

    std::string_view get_name() const

#### `check_against_reference()`

This function is intended to return a boolean which is used by the
benchmarking framework to decide on whether or not the output vector
of minima and maxima shall be compare against the output computed by
Lemire's algorithms.

    bool check_against_reference() const

#### `calc()`

This function is intended to do the main computation task of the
algorithm (including any memory allocation and data transfer between
CPU and GPU (and vice versa) if required by the algorithm). `array` is
hereby a vector of `float`s which serves as an input to the streaming
min/max algorithm. `width` is the width of the sliding window for the
min/max computation.

    void calc(std::vector<float> const & array,	unsigned int width)

#### `get_max_values()`

This function is intended to retrieve the maximum values computed by
the streaming min/max algorithm using the `calc()` function.

    std::vector<float> const & get_max_values() const
    
#### `get_min_values()`

This function is intended to retrieve the minimum values computed by
the streaming min/max algorithm using the `calc()` function.

    std::vector<float> const & get_min_values() const
	
### Streaming min-max algorithm collection

`streaming_min_max_algorithms.h` and
`streaming_min_max_algorithms.cpp` contain a vector of all supported
streaming min/max algorithms (`algorithms`).

    std::vector<std::unique_ptr<streaming_min_max_algorithm_interface> > algorithms

This vector is used by the benchmarking framework to iterate over all
supported algorithms durign one benchmark run.

### Streaming min-max benchmark framework
	
`streaming_min_max_comparison.cpp` contains a simple benchmarking
framework which does the following things:

* handling of command line arguments (see [Running](#Running))
  - size of input vector (`sample_size`)
  - width of sliding window (`window_size`)
  - number of benchmarking iterations (`ìterations`)
* creation of a input vector of `sample_size` random `float`values
* invocation of each algorithm's `calc()` function
* measurement of the run-time of each algorithm's `calc()` function

### Concrete streaming min-max algorithms

#### Lemire's implementation

The files `streaming_min_max_lemire.cpp` and
`streaming_min_max_lemire.h` provide [Lemire's
implementation](https://github.com/lemire/runningmaxmin) of the
algorithm. This implementation is a CPU-only implementation (i.e., the
GPU is *not* involved here).

#### Plain CUDA implementation - explicit memory allocation and transfer - `cudaMalloc()`and `cudaMemCpy()`

The files `streaming_min_max_cuda_plain_malloc.cpp`,
`streaming_min_max_cuda_plain_malloc_cuda.cu`,
`streaming_min_max_cuda_plain_malloc_cuda.cuh` and
`streaming_min_max_cuda_plain_malloc.h` provides a rather simple (and
thus inperformant) GPU implementation using plain
[CUDA](https://developer.nvidia.com/cuda-zone) with `cudaMalloc()`and
`cudaMemCpy()` for allocating and transferring data between host and
device memory.

Hereby the files `streaming_min_max_cuda_plain_malloc_cuda.cu` and
`streaming_min_max_cuda_plain_malloc_cuda.cuh` contain the
implementation of the CUDA kernel and the CUDA-related CPU code (e.g.,
memory allocation, memory transfer, kernel launch) whereas
`streaming_min_max_cuda_plain_malloc.cpp` and
`streaming_min_max_cuda_plain_malloc.h` contain the implementation of
the `streaming_min_max_algorithm_interface` interface for integrate
with the [benchmarking
framework](Streaming-min-max-benchmark-framework).

#### Plain CUDA implementation - page locked memory - `cudaHostRegister()`

The files `streaming_min_max_cuda_plain_page_locked.cpp`,
`streaming_min_max_cuda_plain_page_locked_cuda.cu`,
`streaming_min_max_cuda_plain_page_locked_cuda.cuh` and
`streaming_min_max_cuda_plain_page_locked.h` provides the same simple
(and thus inperformant) GPU implementation using plain
[CUDA](https://developer.nvidia.com/cuda-zone) with page locked host
memory which is registered with `cudaHostRegister()` to allow for
using this memory by the GPU as well.

Hereby the files `streaming_min_max_cuda_plain_page_locked_cuda.cu`
and `streaming_min_max_cuda_plain_page_locked_cuda.cuh` contain the
implementation of the CUDA kernel and the CUDA-related CPU code (e.g.,
memory allocation, memory transfer, kernel launch) whereas
`streaming_min_max_cuda_plain_page_locked.cpp` and
`streaming_min_max_cuda_plain_page_locked.h` contain the
implementation of the `streaming_min_max_algorithm_interface`
interface for integrate with the [benchmarking
framework](Streaming-min-max-benchmark-framework).

#### Plain CUDA implementation - page locked memory with shared cache - `cudaHostRegister()` and ` __shared__`

The files `streaming_min_max_cuda_plain_page_locked_shared.cpp`,
`streaming_min_max_cuda_plain_page_locked_cuda_shared.cu`,
`streaming_min_max_cuda_plain_page_locked_cuda_shared.cuh` and
`streaming_min_max_cuda_plain_page_locked_shared.h` provides almost
the same simple GPU implementation using plain
[CUDA](https://developer.nvidia.com/cuda-zone) with page locked host
memory which is registered with `cudaHostRegister()` to allow for
using this memory by the GPU as well. To prevent having to pay the
performance penalty upon access to the page locked host memory, shared
memory on the device is used a a program controlled cache for all
threads of a thread block (i.e., each element from host memory is only
read *once* into the cache. All subsequent reads take make use of the
cached value in the shared memory.

Hereby the files
`streaming_min_max_cuda_plain_page_locked_shared_cuda.cu` and
`streaming_min_max_cuda_plain_page_locked_shared_cuda.cuh` contain the
implementation of the CUDA kernel and the CUDA-related CPU code (e.g.,
memory allocation, memory transfer, kernel launch) whereas
`streaming_min_max_cuda_plain_page_locked_shared.cpp` and
`streaming_min_max_cuda_plain_page_locked_shared.h` contain the
implementation of the `streaming_min_max_algorithm_interface`
interface for integrate with the [benchmarking
framework](Streaming-min-max-benchmark-framework).

#### CUDA streams implementation

The files `streaming_min_max_cuda_streams.cpp` and
`streaming_min_max_cuda_streams.h` currently only provide provide the
implementation of the `streaming_min_max_algorithm_interface`
interface for integrate with the [benchmarking
framework](#Streaming-min-max-benchmark-framework). - The `calc()`
function is currently (almost) empty. At a later stage of the project
the files are intended to host a [CUDA
streams](https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)
based implementation of the streaming min/max algorithm.

#### Thrust implementation

The files `streaming_min_max_thrust.cpp` and
`streaming_min_max_thrust.h` currently only provide provide the
implementation of the `streaming_min_max_algorithm_interface`
interface for integrate with the [benchmarking
framework](#Streaming-min-max-benchmark-framework). - The `calc()`
function is currently (almost) empty. At a later stage of the project
the files are intended to host a
[Thrust](https://developer.nvidia.com/thrust) based implementation of
the streaming min/max algorithm.

#### Utilities and helpers

The files `utils.cpp` and `utils.h` contains utility functions and
macros for error handling and console tracing.

## Building

Prior to building the executable, the variable `CUDA_PATH`needs to be
adjusted to match the path to the CUDA installation.

On a CUDA equipped machine the executable
`streaming_min_max_comparison` is built by invoking

    make -f Makefile_cuda

for a *release build* and

    make dbg=1 -f Makefile_cuda

for a *debug build*.

Debug builds keep all intermediate files (e.g., assembly and
pre-processed files), define the `DEBUG` macro, and include debug info
in the final executable by using the following command line switches
during compilation:

    -DDEBUG -g -G -save-temps

In addition to the default target (`all`), the following additional
`make` targets are supported:

* `clean` - remove intermediate files (e.g., `*.o` file) of the build process as well as backup files created by the editor but keep the final executable
* `clobber` - like clean, but additionally remove the final executable
* `archive` - like clobber, but additionally creates a ZIP archive `streaming_min_max_comparison.zip` containing all sources and the `Makefile`s.

## Running

The executable of the benchmarking framework (`streaming_min_max_comparison`) has the following synopsis:

    streaming_min_max_comparison [OPTIONS]
	
where `OPTIONS` may be one or multiple of the following:

* `-h`, `--help` - provide some usage information
* `-w`, `--window_size=WINDOW_SIZE` - define the size of the sliding windod for the streaming min/max algorithm in the range of [1, `std::numeric_limits<int>::max()`]
* `-s`, `--sample_size=SAMPLE_SIZE` - define the number of samples (i.e., size of input vector) for the streaming min/max algorithm in the range of [1, `std::numeric_limits<int>::max()`]
* `-i`, `--iterations=ITERATIONS` - define the number iterations (i.e., number of invocation of the `calc()` function of each algorithm) in the range of [1, `std::numeric_limits<int>::max()`]

If the executable was created via a debug build, the following additional option is supported:

* `-v`, `--verbose` - produce verbose tracing output on `stdout`

In the following a sample run of the benchmarking framework is shown:

	[tom@herz GPU_Arch_SS19]$ ./streaming_min_max_comparison 
	
	Performing a comparison using the following parameters:
	  window_size = 500
	  sample_size = 10000000
	  number_of_iterations = 1
	
	  lemire = 2.34238 seconds
	  cuda plain - cuda malloc = 6.57611 seconds
	  cuda plain - page locked memory = 3.08447 seconds
	  cuda plain - page locked shared memory = 0.117122 seconds
	  thrust = 0.243508 seconds
	  cuda streams = 0.0663359 seconds
