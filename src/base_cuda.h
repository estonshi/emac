#ifndef EMAC_BASE_CUDA
#define EMAC_BASE_CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>


// global device variable
int *__mask_gpu;
float *__det_gpu;
float *__w_gpu;
cudaArray *__model_1_gpu;

// texture
texture<float> __tex_01;
texture<float> __tex_mask;
texture<float, cudaTextureType3D, cudaReadModeElementType> __tex_model;

// constant arrays
__constant__ int __pats_gpu[2];
__constant__ float __center_gpu[2];
__constant__ float __rotm_gpu[9];
__constant__ int __vol_len_gpu[1];
__constant__ int __stoprad_gpu[1];


// cuda error checking
#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if( code != cudaSuccess )
	{
		printf("GPUassert : %s %s %d\n", cudaGetErrorString(code), file, line);
		if( abort ) exit(code);
	}
}

// cifft error checking
#define cufftErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cufftResult code, const char *file, int line, bool abort=true)
{
	if( code != CUFFT_SUCCESS )
	{
		printf("GPUassert : %s %s %d\n", "CUFFT FAILED.", file, line);
		if( abort ) exit(code);
	}
}


#endif