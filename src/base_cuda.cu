#include "base_cuda.h"



/* All functions for C modules to call  */

/*  gpu setup & init & utils  */

extern "C" void setDevice(int gpu_id);

extern "C" void gpu_var_init(int det_x, int det_y, float det_center[2], int num_mask_ron[2], 
	int vol_size, int stoprad, int quat_num, float *quaternion, float *ori_det, 
	int *ori_mask, float *init_model_1, float *init_model_2, float *init_merge_w, int ang_corr_bins);

extern "C" void upload_models_to_gpu(float *model_1, float *model_2, int vol_size);

extern "C" void download_model2_from_gpu(float *model_2, int vol_size);

extern "C" void download_currSlice_from_gpu(float *new_slice, int det_x, int det_y);

extern "C" void memcpy_device_pattern_buf(float *pattern, int det_x, int det_y);

extern "C" void memcpy_device_slice_buf(float *myslice, int det_x, int det_y);

extern "C" void free_cuda_all();

extern "C" void cuda_start_event();

extern "C" float cuda_return_time();

/*   slicing and merging   */

extern "C" void get_slice(int quat_index, float *myslice, int BlockSize, int det_x, int det_y, int MASKPIX);

extern "C" void merge_slice(int quat_index, float *myslice, int BlockSize, int det_x, int det_y);

extern "C" void merge_scaling(int GridSize, int BlockSize, float scal_factor);

/*   angular correlation   */

extern "C" void do_angcorr(int partition, float* pat, float *result, int det_x, int det_y, int BlockSize, bool inputType);

extern "C" float comp_angcorr(int partition, float *model_slice_ac, float *pattern_ac, int BlockSize);

/*       likelihood        */

extern "C" float calc_likelihood(float beta, float *model_slice, float *pattern, int det_x, int det_y);

extern "C" void maximization_dot(float *pattern, float prob, int det_x, int det_y, float *new_slice, int BlockSize);

extern "C" void maximization_norm(float scaling_factor, int det_x, int det_y, int BlockSize);




/*********************************************/

/*                Init & utils               */

/*********************************************/


void setDevice(int gpu_id)
{
	cudaError_t custatus;
	custatus = cudaSetDevice(gpu_id);
    if(custatus != cudaSuccess){
        printf("Failed to set Device %d. Exit\n", gpu_id);
        exit(custatus);
    }
}



void gpu_var_init(int det_x, int det_y, float det_center[2], int num_mask_ron[2], 
	int vol_size, int stoprad, int quat_num, float *quaternion, float *ori_det, 
	int *ori_mask, float *init_model_1, float *init_model_2, float *init_merge_w, int ang_corr_bins)
{

	// constant
	int pat_s[] = {det_x, det_y};
	int vols[] = {vol_size};
	int stopr[] = {stoprad};
	cudaMemcpyToSymbol(__vol_len_gpu, vols, sizeof(int));
	cudaMemcpyToSymbol(__pats_gpu, pat_s, sizeof(int)*2);
	cudaMemcpyToSymbol(__center_gpu, det_center, sizeof(float)*2);
	cudaMemcpyToSymbol(__stoprad_gpu, stopr, sizeof(int));
	cudaMemcpyToSymbol(__num_mask_ron_gpu, num_mask_ron, sizeof(int)*2);


	// fft handler
	cufftPlan1d(&__cufft_plan_1d, ang_corr_bins, CUFFT_C2C, ang_corr_bins);


	// mask & det
	cudaMalloc((void**)&__det_gpu, pat_s[0]*pat_s[1]*sizeof(float)*3);
	cudaMemcpy(__det_gpu, ori_det, pat_s[0]*pat_s[1]*sizeof(float)*3, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&__mask_gpu, pat_s[0]*pat_s[1]*sizeof(int));
	cudaMemcpy(__mask_gpu, ori_mask, pat_s[0]*pat_s[1]*sizeof(int), cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, __tex_mask, __mask_gpu, pat_s[0]*pat_s[1]*sizeof(int));


	// quaternion
	cudaMalloc((void**)&__quaternion, quat_num*sizeof(float)*4);
	cudaMemcpy(__quaternion, quaternion, quat_num*sizeof(float)*4, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, __tex_quat, __quaternion, quat_num*sizeof(float)*4);


	// model_1 & merge_w & model_2
	cudaChannelFormatDesc volDesc = cudaCreateChannelDesc<float>();
	cudaExtent volExt = make_cudaExtent((int)vol_size, (int)vol_size, (int)vol_size);
	cudaMemcpy3DParms volParms = {0};
		// model_1
	cudaMalloc3DArray(&__model_1_gpu, &volDesc, volExt);
	volParms.srcPtr = make_cudaPitchedPtr((void*)init_model_1, sizeof(float)*vol_size, (int)vol_size, (int)vol_size);
	volParms.dstArray = __model_1_gpu;
	volParms.extent = volExt;
	volParms.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&volParms);
	cudaBindTextureToArray(__tex_model, __model_1_gpu);
		// model_2
	cudaMalloc((void**)&__model_2_gpu, vol_size*vol_size*vol_size*sizeof(float));
	cudaMemcpy(__model_2_gpu, init_model_2, vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyHostToDevice);
		// merge_w
	cudaMalloc((void**)&__w_gpu, vol_size*vol_size*vol_size*sizeof(float));
	cudaMemcpy(__w_gpu, init_merge_w, vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyHostToDevice);


	// __myslice_device & __mypattern_device
	cudaErrchk(cudaMalloc((void**)&__myslice_device, det_x*det_y*sizeof(float)));
	cudaErrchk(cudaMalloc((void**)&__mypattern_device, det_x*det_y*sizeof(float)));


	// host allocated mapped array
	// store slices
	int size_tmp = ((det_x*det_y+__ThreadPerBlock-1)/__ThreadPerBlock);
	cudaHostAlloc( (void**)&__mapped_array_host, 
					size_tmp*sizeof(float), 
					cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaErrchk(cudaHostGetDevicePointer(&__mapped_array_device, __mapped_array_host, 0));

	// judge : open ac acceleration ?
	if(ang_corr_bins > 0){
		// store angular correlation map
		size_tmp = ((ang_corr_bins*ang_corr_bins+__ThreadPerBlock-1)/__ThreadPerBlock);
		cudaHostAlloc( (void**)&__mapped_array_host_2, 
						size_tmp*sizeof(float), 
						cudaHostAllocWriteCombined | cudaHostAllocMapped);
		cudaErrchk(cudaHostGetDevicePointer(&__mapped_array_device_2, __mapped_array_host_2, 0));

		// angular correlation device buffer
		cudaErrchk(cudaMalloc((void**)&__slice_AC_device, ang_corr_bins*ang_corr_bins*sizeof(float)));
		cudaErrchk(cudaMalloc((void**)&__pattern_AC_device, ang_corr_bins*ang_corr_bins*sizeof(float)));
	}
	else{
		__mapped_array_host_2 = NULL;
		__mapped_array_device_2 = NULL;
		__slice_AC_device = NULL;
		__pattern_AC_device = NULL;
	}



	// Events
	cudaErrchk(cudaEventCreate(&__custart));
    cudaErrchk(cudaEventCreate(&__custop));


	// check error
	cudaErrchk(cudaDeviceSynchronize());
	// change init flag
	__initiated = 1;
}


void upload_models_to_gpu(float *model_1, float *model_2, int vol_size)
{
	// model 1
	cudaUnbindTexture(__tex_model);
	cudaExtent volExt = make_cudaExtent(vol_size, vol_size, vol_size);

	cudaMemcpy3DParms volParms = {0};
	volParms.srcPtr = make_cudaPitchedPtr((void*)model_1, sizeof(float)*vol_size, (int)vol_size, (int)vol_size);
	volParms.dstArray = __model_1_gpu;
	volParms.extent = volExt;
	volParms.kind = cudaMemcpyHostToDevice;

	cudaErrchk(cudaMemcpy3D(&volParms));
	cudaErrchk(cudaBindTextureToArray(__tex_model, __model_1_gpu));


	// model 2
	if(model_2 != NULL){
		cudaErrchk(cudaMemcpy(__model_2_gpu, model_2, vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyHostToDevice));
	}
	else{
		cudaErrchk(cudaMemset(__model_2_gpu, 0, vol_size*vol_size*vol_size*sizeof(float)));
	}
}


void download_model2_from_gpu(float *new_model_2, int vol_size)
{
	cudaErrchk(cudaMemcpy(new_model_2, __model_2_gpu, 
		vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyDeviceToHost));
}


void download_currSlice_from_gpu(float *new_slice, int det_x, int det_y)
{
	cudaErrchk(cudaMemcpy(new_slice, __myslice_device, det_x*det_y*sizeof(float), cudaMemcpyDeviceToHost));
}


void memcpy_device_pattern_buf(float *pattern, int det_x, int det_y)
{
	cudaErrchk(cudaMemcpy(__mypattern_device, pattern, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));
}


void memcpy_device_slice_buf(float *myslice, int det_x, int det_y)
{
	if(myslice != NULL){
		cudaErrchk(cudaMemcpy(__myslice_device, myslice, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));
	}
	else{
		cudaErrchk(cudaMemset(__myslice_device, 0, det_x*det_y*sizeof(float)));
	}
}


void free_cuda_all()
{
	// NOTE
	// __constant__ vars do not need to free
	// __tex_01 & __tex_02 is allocated, bind and freed in user's functions !!!

	// free mask
	cudaFree(__mask_gpu);

	// free det
	cudaUnbindTexture(__tex_mask);
	cudaFree(__det_gpu);

	// free quaternion
	cudaUnbindTexture(__tex_quat);
	cudaFree(__quaternion);

	// free model_1
	cudaUnbindTexture(__tex_model);
	cudaFreeArray(__model_1_gpu);

	// free model_2
	cudaFree(__model_2_gpu);

	// free merge_w
	cudaFree(__w_gpu);

	// free myslice and mypattern buffer
	cudaFree(__myslice_device);
	cudaFree(__mypattern_device);

	// free host allocated mapped memory
	cudaFreeHost(__mapped_array_host);
	if(__mapped_array_host_2 != NULL)
		cudaFreeHost(__mapped_array_host_2);

	// free __angcorr_device
	if(__slice_AC_device != NULL)
		cudaFree(__slice_AC_device);
	if(__pattern_AC_device != NULL)
		cudaFree(__pattern_AC_device);

	// destroy cuda event
	cudaEventDestroy(__custart);
	cudaEventDestroy(__custop);

	// destroy cufft handler
	cufftDestroy(__cufft_plan_1d);

	// reset
	cudaDeviceReset();

	// change init flag
	__initiated = 0;
}



void cuda_start_event()
{
    cudaErrchk(cudaEventRecord(__custart, 0));
}



float cuda_return_time()
{
	float estime;
	cudaErrchk(cudaEventRecord(__custop, 0));
	cudaErrchk(cudaEventSynchronize(__custop));
	cudaErrchk(cudaEventElapsedTime(&estime, __custart, __custop));
	return estime;
}







/*********************************************/

/*                  Merge                    */

/*********************************************/


void get_rotate_matrix(float *quater, float rotm[9]){
	// return rotm[9] ~ index=[00,01,02,10,11,12,20,21,22]

	float q0, q1, q2, q3;

	q0 = quater[0];
	q1 = quater[1];
	q2 = quater[2];
	q3 = quater[3];

	rotm[0] = (1. - 2.*(q2*q2 + q3*q3));
	rotm[1] = 2.*(q1*q2 + q0*q3);
	rotm[2] = 2.*(q1*q3 - q0*q2);
	rotm[3] = 2.*(q1*q2 - q0*q3);
	rotm[4] = (1. - 2.*(q1*q1 + q3*q3));
	rotm[5] = 2.*(q0*q1 + q2*q3);
	rotm[6] = 2.*(q0*q2 + q1*q3);
	rotm[7] = 2.*(q2*q3 - q0*q1);
	rotm[8] = (1. - 2.*(q1*q1 + q2*q2));

	return;
}


__device__ void get_rot_mat(int quat_index, float rotm[9]){
	// return rotm[9] ~ index=[00,01,02,10,11,12,20,21,22]

	float q0, q1, q2, q3;

	q0 = tex1Dfetch(__tex_quat, quat_index*4 + 0);
	q1 = tex1Dfetch(__tex_quat, quat_index*4 + 1);
	q2 = tex1Dfetch(__tex_quat, quat_index*4 + 2);
	q3 = tex1Dfetch(__tex_quat, quat_index*4 + 3);

	rotm[0] = (1. - 2.*(q2*q2 + q3*q3));
	rotm[1] = 2.*(q1*q2 + q0*q3);
	rotm[2] = 2.*(q1*q3 - q0*q2);
	rotm[3] = 2.*(q1*q2 - q0*q3);
	rotm[4] = (1. - 2.*(q1*q1 + q3*q3));
	rotm[5] = 2.*(q0*q1 + q2*q3);
	rotm[6] = 2.*(q0*q2 + q1*q3);
	rotm[7] = 2.*(q2*q3 - q0*q1);
	rotm[8] = (1. - 2.*(q1*q1 + q2*q2));

	return;
}


__global__ void slicing(float *ori_det, float *myslice, int *mymask, int quat_index, int MASKPIX){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// get rotation matrix
	float rotm_cache[9];
	get_rot_mat(quat_index, rotm_cache);

	float q0, q1, q2;
	float d[3];
	float inten = 0, w = 0;

	if(offset >= __pats_gpu[0]*__pats_gpu[1]) return;
	x = mymask[offset];
	if(x > MASKPIX){
		myslice[offset] = 0;
		return;
	}

	// calculate rotated det coordinate
	q0 = ori_det[offset*3];
	q1 = ori_det[offset*3+1];
	q2 = ori_det[offset*3+3];
	d[0] = rotm_cache[0] * q0 + rotm_cache[1] * q1 + rotm_cache[2] * q2;
	d[1] = rotm_cache[3] * q0 + rotm_cache[4] * q1 + rotm_cache[5] * q2;
	d[2] = rotm_cache[6] * q0 + rotm_cache[7] * q1 + rotm_cache[8] * q2;

	// interp
	int lx, ly, lz, center;
	center = (__vol_len_gpu[0]-1)/2;
	d[0] = d[0] + center;
	d[1] = d[1] + center;
	d[2] = d[2] + center;
	lx = (int) floor(d[0]);
	ly = (int) floor(d[1]);
	lz = (int) floor(d[2]);

	// boundary control
	if(lx+1>=__vol_len_gpu[0] || ly+1>=__vol_len_gpu[0] || lz+1>=__vol_len_gpu[0] || lx<0 || ly<0 || lz<0){
		myslice[offset] = 0;
		return;
	}

	//if(lx>=0 && ly>=0 && lz>=0){
		q0 = d[0]-lx;
		q1 = d[1]-ly;
		q2 = d[2]-lz;
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx, ly, lz);
	//}

	//if(lx+1<__vol_len_gpu[0] && ly>=0 && lz>=0){
		q0 = lx+1-d[0];
		q1 = d[1]-ly;
		q2 = d[2]-lz;
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx+1, ly, lz);
	//}

	//if(lx+1<__vol_len_gpu[0] && ly+1<__vol_len_gpu[0] && lz>=0){
		q0 = lx+1-d[0];
		q1 = ly+1-d[1];
		q2 = d[2]-lz;
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx+1, ly+1, lz);
	//}

	//if(lx+1<__vol_len_gpu[0] && ly+1<__vol_len_gpu[0] && lz+1<__vol_len_gpu[0]){
		q0 = lx+1-d[0];
		q1 = ly+1-d[1];
		q2 = lz+1-d[2];
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx+1, ly+1, lz+1);
	//}

	//if(lx+1<__vol_len_gpu[0] && ly>=0 && lz+1<__vol_len_gpu[0]){
		q0 = lx+1-d[0];
		q1 = d[1]-ly;
		q2 = lz+1-d[2];
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx+1, ly, lz+1);
	//}

	//if(lx>=0 && ly+1<__vol_len_gpu[0] && lz+1<__vol_len_gpu[0]){
		q0 = d[0]-lx;
		q1 = ly+1-d[1];
		q2 = lz+1-d[2];
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx, ly+1, lz+1);
	//}

	//if(lx>=0 && ly>=0 && lz+1<__vol_len_gpu[0]){
		q0 = d[0]-lx;
		q1 = d[1]-ly;
		q2 = lz+1-d[2];
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx, ly, lz+1);
	//}

	//if(lx>=0 && ly+1<__vol_len_gpu[0] && lz>=0){
		q0 = d[0]-lx;
		q1 = ly+1-d[1];
		q2 = d[2]-lz;
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx, ly+1, lz);
	//}

	if(w<1e-6)
		myslice[offset] = 0;// / w;
	else
		myslice[offset] = inten / w;


	return;
}



void get_slice(int quat_index, float *myslice, int BlockSize, int det_x, int det_y, int MASKPIX)
{

	gpuInitchk();

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( (det_x+BlockSize-1)/BlockSize, (det_y+BlockSize-1)/BlockSize );

	// slicing
	slicing<<<blocks, threads>>>(__det_gpu, __myslice_device, __mask_gpu, quat_index, MASKPIX);

	// copy back
	if (myslice != NULL)
		cudaErrchk(cudaMemcpy(myslice, __myslice_device, det_x*det_y*sizeof(float), cudaMemcpyDeviceToHost));

}



// anyone who is interested can also optimize it by using shared memory
// I didn't try that cuz its really complex to arrange such a large volume array
// ...
// ok ... cuz I'm lazy ...
__global__ void merging(float *ori_det, float *myslice, int *mymask, float *new_model, float *merge_w, int quat_index)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float q0, q1, q2, tmp, val;
	float d[3];
	long int lx, ly, lz;
	int center;

	if(offset >= __pats_gpu[0]*__pats_gpu[1]) return;
	x = mymask[offset];
	if(x > 1){
		return;
	}

	// quaternion
	float rotm_cache[9];
	get_rot_mat(quat_index, rotm_cache);

	// calculate rotated det coordinate
	q0 = ori_det[offset*3];
	q1 = ori_det[offset*3+1];
	q2 = ori_det[offset*3+3];
	d[0] = rotm_cache[0] * q0 + rotm_cache[1] * q1 + rotm_cache[2] * q2;
	d[1] = rotm_cache[3] * q0 + rotm_cache[4] * q1 + rotm_cache[5] * q2;
	d[2] = rotm_cache[6] * q0 + rotm_cache[7] * q1 + rotm_cache[8] * q2;

	center = (__vol_len_gpu[0]-1)/2;
	d[0] = d[0] + center;
	d[1] = d[1] + center;
	d[2] = d[2] + center;
	lx = (long int) floor(d[0]);
	ly = (long int) floor(d[1]);
	lz = (long int) floor(d[2]);

	val = myslice[offset];

	// boundary control
	if(lx+1>=__vol_len_gpu[0] || ly+1>=__vol_len_gpu[0] || lz+1>=__vol_len_gpu[0] || lx<0 || ly<0 || lz<0){
		return;
	}

	// interp
	x = __vol_len_gpu[0] * __vol_len_gpu[0];
	y = __vol_len_gpu[0];

	//if(lx>=0 && ly>=0 && lz>=0){
		q0 = d[0]-lx;
		q1 = d[1]-ly;
		q2 = d[2]-lz;
		tmp = sqrt( q0*q0 + q1*q1 + q2*q2 );
		tmp = __expf(-tmp/0.3f);
		atomicAdd(&merge_w[lx * x + ly * y + lz], tmp);
		atomicAdd(&new_model[lx * x + ly * y + lz], tmp*val);
	//}

	//if(lx+1<__vol_len_gpu[0] && ly>=0 && lz>=0){
		q0 = lx+1-d[0];
		//q1 = d[1]-ly;
		//q2 = d[2]-lz;
		tmp = sqrt( q0*q0 + q1*q1 + q2*q2 );
		tmp = __expf(-tmp/0.3f);
		atomicAdd(&merge_w[(lx+1) * x + ly * y + lz], tmp);
		atomicAdd(&new_model[(lx+1) * x + ly * y + lz], tmp*val);
	//}

	//if(lx+1<__vol_len_gpu[0] && ly+1<__vol_len_gpu[0] && lz>=0){
		//q0 = lx+1-d[0];
		q1 = ly+1-d[1];
		//q2 = d[2]-lz;
		tmp = sqrt( q0*q0 + q1*q1 + q2*q2 );
		tmp = __expf(-tmp/0.3f);
		atomicAdd(&merge_w[(lx+1) * x + (ly+1) * y + lz], tmp);
		atomicAdd(&new_model[(lx+1) * x + (ly+1) * y + lz], tmp*val);
	//}

	//if(lx+1<__vol_len_gpu[0] && ly+1<__vol_len_gpu[0] && lz+1<__vol_len_gpu[0]){
		//q0 = lx+1-d[0];
		//q1 = ly+1-d[1];
		q2 = lz+1-d[2];
		tmp = sqrt( q0*q0 + q1*q1 + q2*q2 );
		tmp = __expf(-tmp/0.3f);
		atomicAdd(&merge_w[(lx+1) * x + (ly+1) * y + (lz+1)], tmp);
		atomicAdd(&new_model[(lx+1) * x + (ly+1) * y + (lz+1)], tmp*val);
	//}

	//if(lx+1<__vol_len_gpu[0] && ly>=0 && lz+1<__vol_len_gpu[0]){
		//q0 = lx+1-d[0];
		q1 = d[1]-ly;
		//q2 = lz+1-d[2];
		tmp = sqrt( q0*q0 + q1*q1 + q2*q2 );
		tmp = __expf(-tmp/0.3f);
		atomicAdd(&merge_w[(lx+1) * x + ly * y + (lz+1)], tmp);
		atomicAdd(&new_model[(lx+1) * x + ly * y + (lz+1)], tmp*val);
	//}

	//if(lx>=0 && ly+1<__vol_len_gpu[0] && lz+1<__vol_len_gpu[0]){
		q0 = d[0]-lx;
		q1 = ly+1-d[1];
		//q2 = lz+1-d[2];
		tmp = sqrt( q0*q0 + q1*q1 + q2*q2 );
		tmp = __expf(-tmp/0.3f);
		atomicAdd(&merge_w[lx * x + (ly+1) * y + (lz+1)], tmp);
		atomicAdd(&new_model[lx * x + (ly+1) * y + (lz+1)], tmp*val);
	//}

	//if(lx>=0 && ly>=0 && lz+1<__vol_len_gpu[0]){
		//q0 = d[0]-lx;
		q1 = d[1]-ly;
		//q2 = lz+1-d[2];
		tmp = sqrt( q0*q0 + q1*q1 + q2*q2 );
		tmp = __expf(-tmp/0.3f);
		atomicAdd(&merge_w[lx * x + ly * y + (lz+1)], tmp);
		atomicAdd(&new_model[lx * x + ly * y + (lz+1)], tmp*val);
	//}

	//if(lx>=0 && ly+1<__vol_len_gpu[0] && lz>=0){
		//q0 = d[0]-lx;
		q1 = ly+1-d[1];
		q2 = d[2]-lz;
		tmp = sqrt( q0*q0 + q1*q1 + q2*q2 );
		tmp = __expf(-tmp/0.3f);
		atomicAdd(&merge_w[lx * x + (ly+1) * y + lz], tmp);
		atomicAdd(&new_model[lx * x + (ly+1) * y + lz], tmp*val);
	//}

	return;
}



// this function really fucks up. 8 ms to finish on K80 !
// but do we really need to optimize it ?
// it ONLY runs one time at the end of each reconstruction iteration.
__global__ void merging_sc(float *new_model, float *merge_w, float scal_factor)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	long int offset = x + y * blockDim.x * gridDim.x;

	while(offset < __vol_len_gpu[0]*__vol_len_gpu[0]*__vol_len_gpu[0]){
		
		new_model[offset] = new_model[offset] / merge_w[offset] * scal_factor;

		offset += blockDim.x * gridDim.x * blockDim.y * gridDim.y;
	}

}



void merge_slice(int quat_index, float *myslice, int BlockSize, int det_x, int det_y)
{

	gpuInitchk();

	if (myslice != NULL)
		cudaErrchk(cudaMemcpy(__myslice_device, myslice, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( (det_x+BlockSize-1)/BlockSize, (det_y+BlockSize-1)/BlockSize );

	// merging
	merging<<<blocks, threads>>>(__det_gpu, __myslice_device, __mask_gpu, __model_2_gpu, __w_gpu, quat_index);


}



void merge_scaling(int GridSize, int BlockSize, float scal_factor)
{

	gpuInitchk();

	// check scale factor
	if(scal_factor <= 0) scal_factor = 1;

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( GridSize, GridSize );

	// scaling
	merging_sc<<<blocks, threads>>>(__model_2_gpu, __w_gpu, scal_factor);

}







/*********************************************/

/*            Angular Correlation            */

/*********************************************/

__global__ void polar_transfer(cufftReal* polar, int partition)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if(x >= partition || y >= partition) return;

	float xx, yy, phi, r;
	int lx, ly;
	int patx = __pats_gpu[0];
	int paty = __pats_gpu[1];

	phi = (float)x / partition * PI * 2.0f;
	lx = (__pats_gpu[0]>__pats_gpu[1] ? __pats_gpu[1]:__pats_gpu[0]);
	r = (float)y / partition * (lx / 2.0f - __stoprad_gpu[0]) + __stoprad_gpu[0];

	xx = r * __sinf(phi) + __center_gpu[0];  // for 2D matrix, get value with pattern[xx,yy]
	yy = r * __cosf(phi) + __center_gpu[1];

	float inten = 0;
	float w = 0;

	lx = (int)floor(xx);
	ly = (int)floor(yy);
	if(lx>=0 && ly>=0 && tex1Dfetch(__tex_mask, ly + lx*paty)<1){
		r = xx-lx;
		phi = yy-ly;
		r = sqrt( r*r + phi*phi );
		r = __expf(-r/0.3f);
		w += r;
		inten += r * tex1Dfetch(__tex_01, ly + lx*paty);
	}
	if(lx>=0 && ly+1<paty && tex1Dfetch(__tex_mask, ly + 1 + lx*paty)<1){
		r = xx-lx;
		phi = ly+1-yy;
		r = sqrt( r*r + phi*phi );
		r = __expf(-r/0.3f);
		w += r;
		inten += r * tex1Dfetch(__tex_01, ly + 1 + lx*paty);
	}
	if(lx+1<patx && ly>=0 && tex1Dfetch(__tex_mask, ly + (lx+1)*paty)<1){
		r = lx+1-xx;
		phi = yy-ly;
		r = sqrt( r*r + phi*phi );
		r = __expf(-r/0.3f);
		w += r;
		inten += r * tex1Dfetch(__tex_01, ly + (lx+1)*paty);
	}
	if(lx+1<patx && ly+1<paty && tex1Dfetch(__tex_mask, ly + 1 + (lx+1)*paty)<1){
		r = lx+1-xx;
		phi = ly+1-yy;
		r = sqrt( r*r + phi*phi );
		r = __expf(-r/0.3f);
		w += r;
		inten += r * tex1Dfetch(__tex_01, ly + 1 + (lx+1)*paty);
	}

	if(w<1e-6)
		polar[offset] = 0;// / w;
	else
		polar[offset] = inten / w;
}


// transfer cufftReal to cufftComplex
__global__ void cu_Real2Complex(cufftReal* myReal, cufftComplex* mycomplex, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<N){
		mycomplex[i].x = myReal[i];
		mycomplex[i].y = 0;
	}
	return;
}


// calculate complex norm of complex
// 1d grid and 1d block
__global__ void cu_complex_l(cufftComplex* mycomplex, cufftComplex* rets, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<N){
		rets[i].x = mycomplex[i].x * mycomplex[i].x + mycomplex[i].y * mycomplex[i].y;
		rets[i].y = 0;
	}
	return;
}


// calculate real norm of complex
// 1d grid and 1d block
__global__ void cu_complex_r(cufftComplex* mycomplex, cufftReal* rets, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<N){
		rets[i] = sqrt(mycomplex[i].x * mycomplex[i].x + mycomplex[i].y * mycomplex[i].y);
	}
	return;
}


// calculate difference between two angular correlation maps
// Block is 1d and size must be __ThreadPerBlock !!!
// Grid is also 1d and size must be (partition*partition+__ThreadPerBlock-1)/__ThreadPerBlock !!!
// __tex_01 is map1 and __tex_02 is map2
__global__ void angcorr_diff(float *reduced_array, int N)
{
	__shared__ float cache[__ThreadPerBlock];

	int x;
	float k, w;
	int global_offset = threadIdx.x + blockIdx.x * blockDim.x;
	int shared_offset = threadIdx.x;

	if(global_offset >= N){
		cache[shared_offset] = 0;
		return;
	}

	w = tex1Dfetch(__tex_01, global_offset);
	k = tex1Dfetch(__tex_02, global_offset);

	cache[shared_offset] = abs(w - k);

	__syncthreads();

	// reduce
	x = blockDim.x/2;
	while(x != 0)
	{
		if (shared_offset < x)
			cache[shared_offset] += cache[shared_offset + x];
		__syncthreads();
		x /= 2;
	}

	if(shared_offset == 0)
		reduced_array[blockIdx.x] = cache[0];
}




// if "pat" is model slice, set "inputType" = true ;
// else if "pat" is exp pattern, set "inputType" = false .
void do_angcorr(int partition, float* pat, float *result, int det_x, int det_y, int BlockSize, bool inputType)
{
	gpuInitchk();

	// middle var
	cufftComplex *freq_device;
	cudaMalloc((void**)&freq_device, partition*partition*sizeof(cufftComplex));

	// polar grid
	dim3 threads(BlockSize,BlockSize);
	dim3 blocks((partition+BlockSize-1)/BlockSize, (partition+BlockSize-1)/BlockSize);


	// locate device memory & transfer pattern
	if(inputType){

		if(pat != NULL)
			cudaErrchk(cudaMemcpy(__myslice_device, pat, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));

		cudaErrchk(cudaBindTexture(NULL, __tex_01, __myslice_device, det_x*det_y*sizeof(float)));

		// calc polar-coordinate pattern
		polar_transfer<<<blocks, threads>>>(__slice_AC_device, partition);

		// transfer polar_device to cufftComplex
		cu_Real2Complex<<<partition, partition>>>(__slice_AC_device, freq_device, partition*partition);
	}

	else{

		if(pat != NULL)
			cudaErrchk(cudaMemcpy(__mypattern_device, pat, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));

		cudaErrchk(cudaBindTexture(NULL, __tex_01, __mypattern_device, det_x*det_y*sizeof(float)));

		// calc polar-coordinate pattern
		polar_transfer<<<blocks, threads>>>(__pattern_AC_device, partition);

		// transfer polar_device to cufftComplex
		cu_Real2Complex<<<partition, partition>>>(__pattern_AC_device, freq_device, partition*partition);
	}


	// cufft handler has been initiated by gpu_var_init()
	// run fft
	cufftErrchk(cufftExecC2C(__cufft_plan_1d, freq_device, freq_device, CUFFT_FORWARD));

	// self dot
	cu_complex_l<<<partition, partition>>>(freq_device, freq_device, partition*partition);
	
	// run ifft and get angular correlation
	cufftErrchk(cufftExecC2C(__cufft_plan_1d, freq_device, freq_device, CUFFT_INVERSE));


	if(inputType){
		// norm
		cu_complex_r<<<partition, partition>>>(freq_device, __slice_AC_device, partition*partition);
		// return result
		if(result != NULL)
			cudaErrchk(cudaMemcpy(result, __slice_AC_device, partition*partition*sizeof(float), cudaMemcpyDeviceToHost));
	}

	else{
		// norm
		cu_complex_r<<<partition, partition>>>(freq_device, __pattern_AC_device, partition*partition);
		// return result
		if(result != NULL)
			cudaErrchk(cudaMemcpy(result, __pattern_AC_device, partition*partition*sizeof(float), cudaMemcpyDeviceToHost));
	}


	// destroy
	cudaUnbindTexture(__tex_01);
	cudaFree(freq_device);
}



float comp_angcorr(int partition, float *model_slice_ac, float *pattern_ac, int BlockSize)
{
	gpuInitchk();

	if(model_slice_ac != NULL)
		cudaErrchk(cudaMemcpy(__slice_AC_device, model_slice_ac, partition*partition*sizeof(float), cudaMemcpyHostToDevice));
	if(pattern_ac != NULL)
		cudaErrchk(cudaMemcpy(__pattern_AC_device, pattern_ac, partition*partition*sizeof(float), cudaMemcpyHostToDevice));

	// bind texture on __slice_AC_device & __pattern_AC_device
	cudaErrchk(cudaBindTexture(NULL, __tex_01, __slice_AC_device, partition*partition*sizeof(float)));
	cudaErrchk(cudaBindTexture(NULL, __tex_02, __pattern_AC_device, partition*partition*sizeof(float)));

	// reduced array length
	int array_length = (partition*partition+__ThreadPerBlock-1)/__ThreadPerBlock;

	// calc, __mapped_array_device_2 and __mapped_array_host_2 is initiated in gpu_var_init()
	angcorr_diff<<<array_length, __ThreadPerBlock>>> (__mapped_array_device_2, partition*partition);
	cudaErrchk(cudaThreadSynchronize());

	// reduce
	float c = 0;
	int i;
	for(i=0; i<array_length; i++){
		c += __mapped_array_host_2[i];
	}
	c = c/(partition*partition);

	// unbind texture
	cudaUnbindTexture(__tex_01);
	cudaUnbindTexture(__tex_02);

	return c;
}






/*********************************************/

/*         likelihood & maximization         */

/*********************************************/


// Block is 1d and size must be __ThreadPerBlock !!!
// Grid is also 1d and size must be (det_x*det_y+__ThreadPerBlock-1)/__ThreadPerBlock !!!
// __tex_01 is slice from __model_1_gpu and __tex_02 is input pattern
// __tex_mask is mask
// return array (reduced_array) length is blockDim.x
__global__ void possion_likelihood(float beta, float *reduced_array)
{
	__shared__ float cache[__ThreadPerBlock];

	int x ;
	int global_offset = threadIdx.x + blockIdx.x * blockDim.x;
	int shared_offset = threadIdx.x;

	x = tex1Dfetch(__tex_mask, global_offset);
	if(global_offset >= __pats_gpu[0]*__pats_gpu[1] || x > 0){
		cache[shared_offset] = 0;
	}
	else{
		float k, w, s;
		w = tex1Dfetch(__tex_01, global_offset);
		k = tex1Dfetch(__tex_02, global_offset);

		if(w<=0){
			cache[shared_offset] = 0;
		}
		else{
			s = (k*__logf(w)-w)*beta;
			s = s/(__pats_gpu[0]*__pats_gpu[1] - (__num_mask_ron_gpu[0]+__num_mask_ron_gpu[1]));
			cache[shared_offset] = s;
		}
	}

	__syncthreads();

	// reduce
	x = blockDim.x/2;
	while(x != 0)
	{
		if (shared_offset < x)
			cache[shared_offset] += cache[shared_offset + x];
		__syncthreads();
		x /= 2;
	}

	if(shared_offset == 0){
		reduced_array[blockIdx.x] = cache[0];
	}

}


// returned "likelihood" is a pointer to a float number
float calc_likelihood(float beta, float *model_slice, float *pattern, int det_x, int det_y)
{
	gpuInitchk();

	// device memory
	if (model_slice != NULL)
		cudaErrchk(cudaMemcpy(__myslice_device, model_slice, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));
	if (pattern != NULL)
		cudaErrchk(cudaMemcpy(__mypattern_device, pattern, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));

	cudaErrchk(cudaBindTexture(NULL, __tex_01, __myslice_device, det_x*det_y*sizeof(float)));
	cudaErrchk(cudaBindTexture(NULL, __tex_02, __mypattern_device, det_x*det_y*sizeof(float)));

	int array_length = (det_x*det_y+__ThreadPerBlock-1)/__ThreadPerBlock;

	// calc, __mapped_array_device and __mapped_array_host are initiated in gpu_var_init()
	possion_likelihood<<<array_length, __ThreadPerBlock>>> (beta, __mapped_array_device);
	cudaErrchk(cudaThreadSynchronize());

	// reduce
	double c = 0;
	int i;
	for(i=0; i<array_length; i++){
		c += __mapped_array_host[i];
	}
	c = exp(c);

	// unbind texture
	cudaUnbindTexture(__tex_01);
	cudaUnbindTexture(__tex_02);

	return c;

}



// __tex_mask should be initiated in advance
// __tex_01 (__mypattern_device) should be initiated if add == 1
__global__ void pat_dot_float(float number, float *new_slice, int add)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float k;

	x = tex1Dfetch(__tex_mask, offset);
	if(offset >= __pats_gpu[0]*__pats_gpu[1] || x > 1) return;

	if(add){
		k = tex1Dfetch(__tex_01, offset);
		new_slice[offset] += k * number;
	}
	else{
		new_slice[offset] *= number;
	}

}



// maximization procedure should be such a loop :
// {
// 	for r in all_rotation :
//     	memcpy_device_slice_buf (NULL, det_x, det_y)
//      scaling_d = 0
//     	for d in all_data :
//         	p = all_probs[r, d]
//         	maximization_dot (d, p, det_x, det_y, NULL, blocksize)
//			scaling_d += p
//		end
//		maximization_norm (scaling_d, det_x, det_y, blocksize)
//		merge_slice (r, NULL, blocksize, det_x, det_y)
//	end
//	merge_scaling (gridsize, blocksize)
//	download_model2_from_gpu (NEW_MODEL, vol_size)
// }
// more : NEW_MODEL is the returned model


// DO NOT FORGET to initiate __myslice_device to 0 in advance !!!
void maximization_dot(float *pattern, float prob, int det_x, int det_y, float *new_slice, int BlockSize)
{
	gpuInitchk();

	// device memory
	if(pattern != NULL)
		cudaErrchk(cudaMemcpy(__mypattern_device, pattern, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));

	cudaErrchk(cudaBindTexture(NULL, __tex_01, __mypattern_device, det_x*det_y*sizeof(float)));

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( (det_x+BlockSize-1)/BlockSize, (det_y+BlockSize-1)/BlockSize );

	// calc
	pat_dot_float<<<blocks, threads>>>(prob, __myslice_device, 1);

	// unbind texture
	cudaUnbindTexture(__tex_01);

	// copy back ?
	if(new_slice != NULL)
		download_currSlice_from_gpu(new_slice, det_x, det_y);

}


// This function should be called after miximization_dot
void maximization_norm(float scaling_factor, int det_x, int det_y, int BlockSize)
{
	gpuInitchk();

	// device memory
	// we cannot bind texture to __myslice_device, cuz texture memory is read-only

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( (det_x+BlockSize-1)/BlockSize, (det_y+BlockSize-1)/BlockSize );

	// calc
	pat_dot_float<<<blocks, threads>>>(scaling_factor, __myslice_device, 0);

}