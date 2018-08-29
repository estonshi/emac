#!/bin/bash

set -e

cuda_include=$CUDA_HOME/include
cuda_lib=$CUDA_HOME/lib64

nvcc ./src/tst_gpu.cu -o ./bin/tst_gpu -Wno-deprecated-gpu-targets

nvcc -c ./src/base_cuda.cu -o ./src/base_cuda.o --use_fast_math -lcufft -lcufftw -Wno-deprecated-gpu-targets

gcc -I $cuda_include -c ./src/tst_merge.c -o ./src/tst_merge.o

gcc -I $cuda_include -c ./src/tst_ang_corr.c -o ./src/tst_ang_corr.o

gcc ./src/base_cuda.o ./src/tst_merge.o -o ./bin/tst_merge -L $cuda_lib -lcudart -lcufft -lcufftw

gcc ./src/base_cuda.o ./src/tst_ang_corr.o -o ./bin/tst_ang_corr -L $cuda_lib -lcudart -lcufft -lcufftw

rm -rf ./src/base_cuda.o ./src/tst_merge.o ./src/tst_ang_corr.o