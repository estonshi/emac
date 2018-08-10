
cuda_include=$CUDA_HOME/include
cuda_lib=$CUDA_HOME/lib64

nvcc -c ./src/base_cuda.cu -o ./src/base_cuda.o --use_fast_math -lcufft -lcufftw -Wno-deprecated-gpu-targets

gcc -I $cuda_include -c ./src/tst_merge.c -o ./src/tst_merge.o

gcc ./src/base_cuda.o ./src/tst_merge.o -o ./bin/tst_merge -L $cuda_lib -lcudart

rm -rf ./src/base_cuda.o ./src/tst_merge.o