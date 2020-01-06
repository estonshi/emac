#ifndef EMAC_BASE_CUDA
#define EMAC_BASE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "base.h"
#include "emac_data.h"


/* data copy */
int *__dataset_pat_head;         // length = 4*num_pat, [pat0_head_one_loc, pat0_one_num, pat0_head_mul_loc, pat0_mul_num, ...]
                                 // differently in '__dataset_one_loc' and '__dataset_mul_loc' 
int *__dataset_one_loc;       // locations of pixels which have only 1 photon
int *__dataset_mul_loc;       // locations of pixels which have multiple photons
int *__dataset_mul_val;       // number of photons for multi-photon pixels
float *__photon_count;           // photon count for every pattern, length = num_pat
float *__scaling_factor_gpu;         // length = num_pat
__constant__ int __num_dataset[3];    // [num_pat, num_one_photon, num_mul_photon]
texture<int> __tex_dataset_pat_head;
/* data copy */


// global device variable
int *__mask_gpu;
float *__det_gpu;
float *__correction_gpu;
float *__w_gpu;
cudaArray *__model_1_gpu;
float *__model_2_gpu;
cufftHandle __cufft_plan_1d;
float *__quaternion;

// global device buffer memory
float *__myslice_device;
float *__mypattern_device;
float *__slice_AC_device, *__pattern_AC_device;
// (they are all reduced 1d array, only take a little memory)
// __mapped_array_device & __mapped_array_device_2 are buffers for reduce calculation
float *__mapped_array_device;  // length is det_x*det_y*num_threads
float *__mapped_array_device_2;


// texture
texture<float> __tex_01;
texture<float> __tex_02;
texture<int> __tex_mask;
texture<float, cudaTextureType3D, cudaReadModeElementType> __tex_model;
texture<float> __tex_quat;
texture<float> __tex_det;
texture<float> __tex_correction;

// constant arrays
__constant__ int __pats_gpu[2];
__constant__ float __center_gpu[2];
//__constant__ float __rotm_gpu[9];
__constant__ int __vol_len_gpu[1];
__constant__ int __stoprad_gpu[1];
__constant__ int __num_mask_ron_gpu[2];

//const value, must be 2^n
const int __ThreadPerBlock = 128;


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