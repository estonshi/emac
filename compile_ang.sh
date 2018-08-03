 nvcc ./src/tst_ang_corr.cu -o ./bin/tst_ang_corr -Wno-deprecated-gpu-targets --use_fast_math -lcufft -lcufftw
