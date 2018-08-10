#ifndef EMAC_BASE_CUDA
#define EMAC_BASE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "base.h"

// global device variable
int *__mask_gpu;
float *__det_gpu;
cudaArray *__w_gpu;
cudaArray *__model_1_gpu;
cudaArray *__model_2_gpu;

// texture
texture<float> __tex_01;
texture<float> __tex_02;
texture<int> __tex_mask;
texture<float, cudaTextureType3D, cudaReadModeElementType> __tex_model;

// constant arrays
__constant__ int __pats_gpu[2];
__constant__ float __center_gpu[2];
__constant__ float __rotm_gpu[9];
__constant__ int __vol_len_gpu[1];
__constant__ int __stoprad_gpu[1];


// cuda event
cudaEvent_t __custart, __custop;


// initial flag
int __initiated = 0;


// cuda error checking
#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if( code != cudaSuccess )
	{
		printf("GPU-assert : %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

// cufft error checking
#define cufftErrchk(ans) { gpuFFTAssert((ans), __FILE__, __LINE__); }
inline void gpuFFTAssert(cufftResult code, const char *file, int line)
{
	if( code != CUFFT_SUCCESS )
	{
		printf("GPU-FFT-assert : %s %s %d\n", "CUFFT FAILED.", file, line);
		exit(code);
	}
}


// check whether GPU has been initiated
#define gpuInitchk() { gpuInitAssert(__FILE__, __LINE__); }
inline void gpuInitAssert(const char *file, int line)
{
	if( !__initiated )
	{
		printf("GPU-Init-assert : %s %s %d\n", "GPU variables are not initiated.", file, line);
		exit(code);
	}
}



#endif