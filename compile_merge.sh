nvcc ./src/tst_merge.cu -o ./bin/tst_merge -Wno-deprecated-gpu-targets --use_fast_math -lcufft -lcufftw