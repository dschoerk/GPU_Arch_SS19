SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%PATH%
:: ;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin

::-Xcompiler "/std:c++20" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin" 
nvcc -I. -I"C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\inc" -c streaming_min_max_cuda_plain_cuda.cu streaming_min_max_thrust_cuda.cu

nvcc -Xcompiler "/std:c++17" -o minmax.exe -I. -I"C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\inc" -I"C:\Program Files\NVIDIA Corporation\NvToolsExt\include" -L"C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/" -lnvToolsExt64_1 streaming_min_max_cuda_plain_cuda.obj streaming_min_max_thrust_cuda.obj streaming_min_max_algorithms.cpp streaming_min_max_comparison.cpp streaming_min_max_cuda_plain.cpp streaming_min_max_cuda_streams.cpp streaming_min_max_lemire.cpp utils.cpp streaming_min_max_thrust.cpp getopt.cpp

:: -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.20.27508\bin\Hostx64\x64"