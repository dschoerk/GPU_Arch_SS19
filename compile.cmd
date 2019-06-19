SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%PATH%
:: ;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin

::-Xcompiler "/std:c++20" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin" 
nvcc -rdc=true --gpu-architecture=compute_70 --gpu-code=compute_70 -I. -I"C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\inc" -c streaming_min_max_cuda_plain_malloc_cuda.cu streaming_min_max_cuda_plain_page_locked_cuda.cu streaming_min_max_cuda_plain_page_locked_shared_cuda.cu streaming_min_max_thrust_cuda.cu

nvcc -rdc=true --gpu-architecture=compute_70 --gpu-code=compute_70 -Xcompiler "/std:c++17" -o minmax.exe -I. -I"C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\inc" -I"C:\Program Files\NVIDIA Corporation\NvToolsExt\include" -L"C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/" -lnvToolsExt64_1 streaming_min_max_cuda_plain_malloc_cuda.obj streaming_min_max_cuda_plain_page_locked_cuda.obj streaming_min_max_cuda_plain_page_locked_shared_cuda.obj streaming_min_max_thrust_cuda.obj streaming_min_max_algorithms.cpp streaming_min_max_comparison.cpp streaming_min_max_cuda_plain_malloc.cpp streaming_min_max_cuda_plain_page_locked_shared.cpp streaming_min_max_cuda_plain_page_locked.cpp streaming_min_max_cuda_streams.cpp streaming_min_max_lemire.cpp utils.cpp streaming_min_max_thrust.cpp getopt.cpp

:: -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.20.27508\bin\Hostx64\x64"