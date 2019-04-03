#!/bin/bash

cuda_include=$CUDA_HOME/include
cuda_lib=$CUDA_HOME/lib64

# mkdir
if [ ! -d "./bin" ];then
	mkdir ./bin
fi

## compile python files

cp ./py_src/map_det.py ./bin/emac.map_det
chmod u+x ./bin/emac.map_det

cp ./py_src/make_emac_data.py ./bin/emac.make_data
chmod u+x ./bin/emac.make_data

cp ./py_src/read_emac_data.py ./bin/emac.read_data
chmod u+x ./bin/emac.read_data


## compile support C files

nvcc ./src/tst_gpu.cu -o ./bin/emac.gpu_fetch -Wno-deprecated-gpu-targets

nvcc -c ./src/base_cuda.cu -o ./src/base_cuda.o --use_fast_math -lcufft -lcufftw -Wno-deprecated-gpu-targets

gcc -c ./src/emac_data.c -o ./src/emac_data.o -lm

gcc -c ./src/gen_quat.c -o ./src/gen_quat.o -lm

gcc -c ./src/params.c -o ./src/params.o -lm

##gcc -fopenmp -c ./src/main.c -o ./src/main.o -lm
mpicc -fopenmp -c ./src/main.c -o ./src/main.o -lm -O3

## compile test C files

gcc -c ./src/gen_quat_main.c -o ./src/gen_quat_main.o -lm

gcc -c ./src/tst_emac_data.c -o ./src/tst_emac_data.o -lm

gcc -c ./src/tst_params.c -o ./src/tst_params.o -lm

gcc -c ./src/tst_merge.c -o ./src/tst_merge.o

gcc -c ./src/tst_ang_corr.c -o ./src/tst_ang_corr.o

gcc -c ./src/tst_likelihood.c -o ./src/tst_likelihood.o


## link .o files

gcc ./src/gen_quat_main.o ./src/gen_quat.o -o ./bin/emac.gen_quat -lm

gcc ./src/tst_emac_data.o ./src/emac_data.o -o ./bin/emac.tst.emac_data -lm

gcc ./src/tst_params.o ./src/params.o ./src/gen_quat.o ./src/emac_data.o -o ./bin/emac.tst.set_up -lm

gcc ./src/base_cuda.o ./src/tst_merge.o -o ./bin/emac.tst.slice_merge -L $cuda_lib -lcudart -lcufft -lcufftw -lm

gcc ./src/base_cuda.o ./src/tst_ang_corr.o -o ./bin/emac.tst.ang_corr -L $cuda_lib -lcudart -lcufft -lcufftw -lm

gcc ./src/base_cuda.o ./src/emac_data.o ./src/tst_likelihood.o -o ./bin/emac.tst.maximization -L $cuda_lib -lcudart -lcufft -lcufftw -lm

#gcc -fopenmp ./src/base_cuda.o ./src/emac_data.o ./src/gen_quat.o ./src/params.o ./src/main.o -o ./bin/emac.main -L $cuda_lib -lcudart -lcufft -lcufftw -lm
mpicc -fopenmp ./src/base_cuda.o ./src/emac_data.o ./src/gen_quat.o ./src/params.o ./src/main.o -o ./bin/emac.main -L $cuda_lib -lcudart -lcufft -lcufftw -lm -O3

## remove .o files

rm -rf ./src/base_cuda.o ./src/emac_data.o ./src/gen_quat.o ./src/params.o ./src/main.o

rm -rf ./src/gen_quat_main.o ./src/tst_emac_data.o ./src/tst_params.o ./src/tst_merge.o ./src/tst_ang_corr.o ./src/tst_likelihood.o