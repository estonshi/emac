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
float *__w_gpu;
cudaArray *__model_1_gpu;
float *__model_2_gpu;
cufftHandle __cufft_plan_1d;

// global device buffer memory
float *__myslice_device;
float *__mypattern_device;
float *__mapped_array_device, *__mapped_array_host;


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
__constant__ int __num_mask_ron_gpu[2];

//const value
const int __ThreadPerBlock = 256;


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
		printf("GPU-FFT-assert : %s %s %d\n", "CUFFT failed.", file, line);
		exit(code);
	}
}


// check whether GPU has been initiated
#define gpuInitchk() { gpuInitAssert(__FILE__, __LINE__); }
inline void gpuInitAssert(const char *file, int line)
{
	if( !__initiated )
	{
		printf("GPU-Init-assert : %s %s %d\n", "GPU environment hasn't been initiated.", file, line);
		exit(1);
	}
}

/*
// debug
size_t free_byte, total_byte;
cudaErrchk(cudaMemGetInfo(&free_byte, &total_byte));
printf("free mem : %f MB, total mem : %f MB\n", (float)free_byte/1024/1024, (float)total_byte/1024/1024);
*/


#endif