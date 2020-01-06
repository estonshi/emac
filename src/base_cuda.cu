#include "base_cuda.h"



/* All functions for C modules to call  */

/*  gpu setup & var init & utils  */

extern "C" void setDevice(int gpu_id);

extern "C" void gpu_var_init(int det_x, int det_y, float det_center[2], int num_mask_ron[2], 
	int vol_size, int stoprad, int quat_num, float *quaternion, float *ori_det, float *correction, 
	int *ori_mask, float *init_model_1, float *init_model_2, float *init_merge_w, int ang_corr_bins, int num_threads);

extern "C" void upload_models_to_gpu(float *model_1, float *model_2, float *merge_w, int vol_size);

extern "C" void download_model2_from_gpu(float *model_2, int vol_size);

extern "C" void download_volume_from_gpu(float *vol_container, int vol_size, int which);

extern "C" void download_currSlice_from_gpu(float *new_slice, int det_x, int det_y);

extern "C" void download_scaling_from_gpu(float *scaling_factor, int num_data);

extern "C" void reset_model_gpu(int vol_size, int which);

extern "C" void memcpy_device_pattern_buf(float *pattern, int det_x, int det_y);

extern "C" void memcpy_device_slice_buf(float *myslice, int det_x, int det_y);

extern "C" void memcpy_device_scaling_buf(float *scaling_factor, int num_data);

extern "C" void free_device_all();

extern "C" void cuda_start_event();

extern "C" float cuda_return_time();

/*   dataset transfer      */

extern "C" void gpu_dataset_init(emac_pat *dataset, int num_pat);

extern "C" void parse_pattern_ongpu(int inx, int photon_len, int det_x, int det_y, float *mypattern, bool scaling);

extern "C" void free_cuda_dataset();

extern "C" float slice_sum(float *model_slice, int det_x, int det_y, float *input_array);

/*   slicing and merging   */

extern "C" void get_slice(int quat_index, float *myslice, int BlockSize, int det_x, int det_y, int MASKPIX);

extern "C" void merge_slice(int quat_index, float *myslice, int BlockSize, int det_x, int det_y);

extern "C" void merge_scaling(int GridSize, int BlockSize, float scal_factor);

/*   angular correlation   */

extern "C" void do_angcorr(int partition, float* pat, float *result, int det_x, int det_y, int BlockSize, bool inputType);

extern "C" float comp_angcorr(int partition, float *model_slice_ac, float *pattern_ac, int BlockSize);

/*       likelihood        */

extern "C" double calc_likelihood(float beta, float *model_slice, float *pattern, int det_x, int det_y, float scaling_factor);

extern "C" double calc_likelihood_part(int pat_inx, int photon_len, int det_x, int det_y, float beta, float *model_slice, float scaling_factor);

extern "C" double calc_likelihood_part_par(int pat_inx, int photon_len, int det_x, int det_y, float beta, 
												float *model_slice, float scaling_factor, int thread_id);

extern "C" void maximization_dot(float *pattern, float prob, int det_x, int det_y, float *new_slice, int BlockSize);

extern "C" void maximization_dot_part(int pat_inx, float prob, int one_pix_count, int mul_pix_count);

extern "C" void maximization_norm(float scaling_factor, int det_x, int det_y, int BlockSize);

extern "C" void update_scaling(int num_data);


/*********************************************/

/*                Init & utils               */

/*********************************************/

// reset __w_gpu
__global__ void reset_w_gpu(float *merge_w, float value)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	long int offset = x + y * blockDim.x * gridDim.x;

	while(offset < __vol_len_gpu[0]*__vol_len_gpu[0]*__vol_len_gpu[0]){
		
		merge_w[offset] = value;

		offset += blockDim.x * gridDim.x * blockDim.y * gridDim.y;
	}
}


void setDevice(int gpu_id)
{
	cudaError_t custatus;
	custatus = cudaSetDevice(gpu_id);
    if(custatus != cudaSuccess){
        printf("Failed to set Device %d. Exit\n", gpu_id);
        exit(custatus);
    }
}


void gpu_dataset_init(emac_pat *dataset, int num_pat)
{
	// Call after gpu_var_init(...) !!!
	// transfer dataset to GPU ?
	// make __dataset_one_loc, .. etc as small empty array
	if(dataset == NULL){
		cudaMalloc((void**)&__dataset_pat_head, 4*sizeof(int));
		cudaMalloc((void**)&__dataset_one_loc, 2*sizeof(int));
		cudaMalloc((void**)&__dataset_mul_loc, 2*sizeof(int));
		cudaMalloc((void**)&__dataset_mul_val, 2*sizeof(int));
		cudaMalloc((void**)&__scaling_factor_gpu, 2*sizeof(float));
		cudaBindTexture(NULL, __tex_dataset_pat_head, __dataset_pat_head, 4*sizeof(int));
		return;
	}

	free_cuda_dataset();

	emac_pat *pat_struct;
	int total_one_photon = 0, total_mul_photon = 0;
	int count_buff = 0, i=0, j=0;

	// malloc __dataset_pat_loc
	pat_struct = dataset;
	int *dataset_pat_head = (int*) malloc(4*num_pat*sizeof(int));
	for(i=0; i<num_pat; i++){
		dataset_pat_head[4*i] = total_one_photon;
		dataset_pat_head[4*i+1] = pat_struct->one_pix;
		dataset_pat_head[4*i+2] = total_mul_photon;
		dataset_pat_head[4*i+3] = pat_struct->mul_pix;
		total_one_photon += pat_struct->one_pix;
		total_mul_photon += pat_struct->mul_pix;
		pat_struct = pat_struct->next;
	}
	cudaMalloc((void**)&__dataset_pat_head, 4*num_pat*sizeof(int));
	cudaMemcpy(__dataset_pat_head, dataset_pat_head, 4*num_pat*sizeof(int), cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, __tex_dataset_pat_head, __dataset_pat_head, 4*num_pat*sizeof(int));
	free(dataset_pat_head);

	// malloc __dataset_one_loc
	pat_struct = dataset;
	int *dataset_one_loc = (int*) malloc(total_one_photon*sizeof(int));
	count_buff = 0; total_one_photon = 0;
	for(i=0; i<num_pat; i++){
		count_buff = pat_struct->one_pix;
		for(j=0; j<count_buff; j++)
			dataset_one_loc[total_one_photon + j] = pat_struct->one_loc[j];
		total_one_photon += count_buff;
		pat_struct = pat_struct->next;
	}
	cudaMalloc((void**)&__dataset_one_loc, total_one_photon*sizeof(int));
	cudaMemcpy(__dataset_one_loc, dataset_one_loc, total_one_photon*sizeof(int), cudaMemcpyHostToDevice);
	free(dataset_one_loc);

	// malloc __dataset_mul_loc
	pat_struct = dataset;
	int *dataset_mul_loc = (int*) malloc(total_mul_photon*sizeof(int));
	int *dataset_mul_val = (int*) malloc(total_mul_photon*sizeof(int));
	count_buff = 0; total_mul_photon = 0;
	for(i=0; i<num_pat; i++){
		count_buff = pat_struct->mul_pix;
		for(j=0; j<count_buff; j++){
			dataset_mul_loc[total_mul_photon + j] = pat_struct->mul_loc[j];
			dataset_mul_val[total_mul_photon + j] = pat_struct->mul_counts[j];
		}
		total_mul_photon += count_buff;
		pat_struct = pat_struct->next;
	}
	cudaMalloc((void**)&__dataset_mul_loc, total_mul_photon*sizeof(int));
	cudaMemcpy(__dataset_mul_loc, dataset_mul_loc, total_mul_photon*sizeof(int), cudaMemcpyHostToDevice);
	free(dataset_mul_loc);
	cudaMalloc((void**)&__dataset_mul_val, total_mul_photon*sizeof(int));
	cudaMemcpy(__dataset_mul_val, dataset_mul_val, total_mul_photon*sizeof(int), cudaMemcpyHostToDevice);
	free(dataset_mul_val);

	// malloc __scaling_factor_gpu
	pat_struct = dataset;
	float *scaling_factor = (float*) malloc(num_pat*sizeof(float));
	for(i=0; i<num_pat; i++){
		scaling_factor[i] = pat_struct->scale_factor;
		pat_struct = pat_struct->next;
	}
	cudaMalloc((void**)&__scaling_factor_gpu, num_pat*sizeof(float));
	cudaMemcpy(__scaling_factor_gpu, scaling_factor, num_pat*sizeof(float), cudaMemcpyHostToDevice);
	free(scaling_factor);

	// malloc __photon_count
	pat_struct = dataset;
	float *photon_count = (float*) malloc(num_pat*sizeof(float));
	for(i=0; i<num_pat; i++){
		photon_count[i] = pat_struct->photon_count;
		pat_struct = pat_struct->next;
	}
	cudaMalloc((void**)&__photon_count, num_pat*sizeof(float));
	cudaMemcpy(__photon_count, photon_count, num_pat*sizeof(float), cudaMemcpyHostToDevice);
	free(photon_count);

	// initiate __num_pattern, __tex_dataset_pat_head
	int tmp[3];
	tmp[0] = num_pat; tmp[1] = total_one_photon; tmp[2] = total_mul_photon;
	cudaMemcpyToSymbol(__num_dataset, tmp, sizeof(int)*3);
	pat_struct = NULL;
	
}



void gpu_var_init(int det_x, int det_y, float det_center[2], int num_mask_ron[2], int vol_size, 
	int stoprad, int quat_num, float *quaternion, float *ori_det, float *correction, int *ori_mask, 
	float *init_model_1, float *init_model_2, float *init_merge_w, int ang_corr_bins, int num_threads)
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
	cudaBindTexture(NULL, __tex_det, __det_gpu, pat_s[0]*pat_s[1]*sizeof(float)*3);

	cudaMalloc((void**)&__correction_gpu, pat_s[0]*pat_s[1]*sizeof(float));
	cudaMemcpy(__correction_gpu, correction, pat_s[0]*pat_s[1]*sizeof(float), cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, __tex_correction, __correction_gpu, pat_s[0]*pat_s[1]*sizeof(float));

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


	// Buffer for reduce calculation
	// store slices
	int size_tmp = (det_x*det_y+__ThreadPerBlock-1)/__ThreadPerBlock;
	cudaErrchk(cudaMalloc((void**)&__mapped_array_device, num_threads*size_tmp*sizeof(float)));

	// judge : open ac acceleration ?
	if(ang_corr_bins > 0){
		// store angular correlation map
		size_tmp = ((ang_corr_bins*ang_corr_bins+__ThreadPerBlock-1)/__ThreadPerBlock);
		cudaErrchk(cudaMalloc((void**)&__mapped_array_device_2, size_tmp*sizeof(float)));

		// angular correlation device buffer
		cudaErrchk(cudaMalloc((void**)&__slice_AC_device, ang_corr_bins*ang_corr_bins*sizeof(float)));
		cudaErrchk(cudaMalloc((void**)&__pattern_AC_device, ang_corr_bins*ang_corr_bins*sizeof(float)));
	}
	else{
		__mapped_array_device_2 = NULL;
		__slice_AC_device = NULL;
		__pattern_AC_device = NULL;
	}

	// Events
	cudaErrchk(cudaEventCreate(&__custart));
    cudaErrchk(cudaEventCreate(&__custop));


    // datasets empty initiation
    gpu_dataset_init(NULL, 0);

	// check error
	cudaErrchk(cudaDeviceSynchronize());
	// change init flag
	__initiated = 1;
}


void upload_models_to_gpu(float *model_1, float *model_2, float *merge_w, int vol_size)
{
	// model 1
	if(model_1 != NULL){
		cudaUnbindTexture(__tex_model);
		cudaExtent volExt = make_cudaExtent(vol_size, vol_size, vol_size);

		cudaMemcpy3DParms volParms = {0};
		volParms.srcPtr = make_cudaPitchedPtr((void*)model_1, sizeof(float)*vol_size, (int)vol_size, (int)vol_size);
		volParms.dstArray = __model_1_gpu;
		volParms.extent = volExt;
		volParms.kind = cudaMemcpyHostToDevice;

		cudaErrchk(cudaMemcpy3D(&volParms));
		cudaErrchk(cudaBindTextureToArray(__tex_model, __model_1_gpu));
	}

	// model 2
	if(model_2 != NULL){
		cudaErrchk(cudaMemcpy(__model_2_gpu, model_2, vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyHostToDevice));
	}

	// merge_w
	if(merge_w != NULL){
		cudaErrchk(cudaMemcpy(__w_gpu, merge_w, vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyHostToDevice));
	}

}


void download_model2_from_gpu(float *new_model_2, int vol_size)
{
	cudaErrchk(cudaMemcpy(new_model_2, __model_2_gpu, 
		vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyDeviceToHost));
}


void download_volume_from_gpu(float *vol_container, int vol_size, int which)
{
	if(which == 0){
		cudaErrchk(cudaMemcpy(vol_container, __w_gpu, vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else if(which == 1){
		cudaErrchk(cudaMemcpy(vol_container, __model_1_gpu, vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else if(which == 2){
		cudaErrchk(cudaMemcpy(vol_container, __model_2_gpu, vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else{
		vol_container = NULL;
	}
}


void download_currSlice_from_gpu(float *new_slice, int det_x, int det_y)
{
	cudaErrchk(cudaMemcpy(new_slice, __myslice_device, det_x*det_y*sizeof(float), cudaMemcpyDeviceToHost));
}


void download_scaling_from_gpu(float *scaling_factor, int num_data)
{
	cudaErrchk(cudaMemcpy(scaling_factor, __scaling_factor_gpu, num_data*sizeof(float), cudaMemcpyDeviceToHost));
}


void reset_model_gpu(int vol_size, int which){
	if(which == 0){
		/*
		dim3 threads( 16, 16 );
		dim3 blocks( 16, 16 );
		reset_w_gpu<<<blocks, threads>>>(__w_gpu, 1.0f);
		*/
		cudaErrchk(cudaMemset(__w_gpu, 0, vol_size*vol_size*vol_size*sizeof(float)));
	}
	else if(which == 1){
		cudaErrchk(cudaMemset(__model_1_gpu, 0, vol_size*vol_size*vol_size*sizeof(float)));
	}
	else if(which == 2){
		cudaErrchk(cudaMemset(__model_2_gpu, 0, vol_size*vol_size*vol_size*sizeof(float)));
	}
	else{
		;
	}
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


void memcpy_device_scaling_buf(float *scaling_factor, int num_data)
{
	cudaErrchk(cudaMemcpy(__scaling_factor_gpu, scaling_factor, num_data*sizeof(float), cudaMemcpyHostToDevice));
}


void free_cuda_dataset()
{
	cudaFree(__dataset_one_loc);
	cudaFree(__dataset_mul_loc);
	cudaFree(__dataset_mul_val);
	cudaFree(__scaling_factor_gpu);
	cudaUnbindTexture(__tex_dataset_pat_head);
	cudaFree(__dataset_pat_head);
}


void free_device_all()
{
	// NOTE
	// __constant__ vars do not need to free
	// __tex_01 & __tex_02 is allocated, bind and freed in user's functions !!!

	// free dataset
	free_cuda_dataset();

	// free mask
	cudaUnbindTexture(__tex_mask);
	cudaFree(__mask_gpu);

	// free det
	cudaUnbindTexture(__tex_det);
	cudaUnbindTexture(__tex_correction);
	cudaFree(__det_gpu);
	cudaFree(__correction_gpu);

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

	// free buffer
	cudaFree(__mapped_array_device);
	if(__mapped_array_device_2 != NULL)
		cudaFree(__mapped_array_device_2);

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

/*                 Dataset                   */

/*********************************************/


// Block size is suggested to use __ThreadPerBlock
// Grid size is (one_pix_num + mul_pix_num + __ThreadPerBlock -1) / __ThreadPerBlock
// new_pat should be memset to 0 !!!
__global__ void do_parse_pattern(int inx, float *new_pat, int *dataset_one_loc, int *dataset_mul_loc, 
								int *dataset_mul_val)
{

	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	
	// get locations
	int one_head, mul_head, one_len, mul_len, loc_tmp;
	float correction;
	one_head = tex1Dfetch(__tex_dataset_pat_head, 4*inx);
	one_len  = tex1Dfetch(__tex_dataset_pat_head, 4*inx+1);
	mul_head = tex1Dfetch(__tex_dataset_pat_head, 4*inx+2);
	mul_len  = tex1Dfetch(__tex_dataset_pat_head, 4*inx+3);
	correction = tex1Dfetch(__tex_correction, inx);

	if( offset >= one_len + mul_len ) return;

	if( offset < one_len ){
		loc_tmp = dataset_one_loc[one_head + offset];
		new_pat[loc_tmp] = 1 / correction;
	}
	else{
		loc_tmp = dataset_mul_loc[mul_head + offset - one_len];
		new_pat[loc_tmp] = dataset_mul_val[mul_head + offset - one_len] / correction;
	}
}


// photon_len = one_pix_num + mul_pix_num
void parse_pattern_ongpu(int inx, int photon_len, int det_x, int det_y, float *mypattern, bool scaling)
{

	gpuInitchk();
	cudaErrchk(cudaMemset(__mypattern_device, 0, det_x*det_y*sizeof(float)));

	int array_length = (photon_len+__ThreadPerBlock-1)/__ThreadPerBlock;

	// parse pattern
	do_parse_pattern<<<array_length, __ThreadPerBlock>>> (inx, __mypattern_device, 
			__dataset_one_loc, __dataset_mul_loc, __dataset_mul_val);

	// return ?
	if(mypattern != NULL){
		cudaErrchk(cudaMemcpy(mypattern, __mypattern_device, det_x*det_y*sizeof(float), cudaMemcpyDeviceToHost));
	}

}


// device code for reduce calculation
__device__ void warpReduce(volatile float *sdata, int tid)
{
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce(float *input_array, float *output_value, unsigned int n)
{

	__shared__ float sdata[__ThreadPerBlock];

	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	int index_offset = index + blockDim.x;
	int s;

	if(index >=n ) sdata[tid] = 0;
	else if (index_offset >= n) sdata[tid] = input_array[index];
	else sdata[tid] = input_array[index] + input_array[index_offset];

	__syncthreads();

	for(s=blockDim.x/2; s>32; s>>=1){
		if(tid < s) sdata[tid] += sdata[tid+s];
		__syncthreads();
	}

	if(tid < 32) warpReduce(sdata, tid);

	if(tid == 0) output_value[blockIdx.x] = sdata[0];

}

float slice_sum(float *model_slice, int det_x, int det_y, float *input_array)
{

	gpuInitchk();

	float *this_input_array;
	if(input_array == NULL) this_input_array = __myslice_device;
	else this_input_array = input_array;

	int i, blocks, iNum;
	int size = det_x * det_y;
	float *value = (float*)malloc(sizeof(float));
	if(model_slice != NULL){
		cudaErrchk(cudaMemcpy(this_input_array, model_slice, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));
	}

	for(i=1, iNum=size; i<size; i=i*2*__ThreadPerBlock){
		blocks = (iNum + 2*__ThreadPerBlock - 1) / (2*__ThreadPerBlock);
		reduce<<<blocks, __ThreadPerBlock>>> (this_input_array, this_input_array, iNum);
		iNum = blocks;
	}

	cudaMemcpy(value, this_input_array, sizeof(float), cudaMemcpyDeviceToHost);
	
	return value[0];

}

float slice_sum_par(float *model_slice, int det_x, int det_y, float *input_array, cudaStream_t stream)
{

	gpuInitchk();

	float *this_input_array;
	if(input_array == NULL) this_input_array = __myslice_device;
	else this_input_array = input_array;

	int i, blocks, iNum;
	int size = det_x * det_y;
	float *value = (float*)malloc(sizeof(float));
	if(model_slice != NULL){
		cudaErrchk(cudaMemcpy(this_input_array, model_slice, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));
	}

	for(i=1, iNum=size; i<size; i=i*2*__ThreadPerBlock){
		blocks = (iNum + 2*__ThreadPerBlock - 1) / (2*__ThreadPerBlock);
		reduce<<<blocks, __ThreadPerBlock, 0, stream>>> (this_input_array, this_input_array, iNum);
		iNum = blocks;
	}

	cudaMemcpy(value, this_input_array, sizeof(float), cudaMemcpyDeviceToHost);
	
	return value[0];

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
	rotm[1] = 2.*(q1*q2 - q0*q3);
	rotm[2] = 2.*(q1*q3 + q0*q2);
	rotm[3] = 2.*(q1*q2 + q0*q3);
	rotm[4] = (1. - 2.*(q1*q1 + q3*q3));
	rotm[5] = 2.*(q2*q3 - q0*q1);
	rotm[6] = 2.*(q1*q3 - q0*q2);
	rotm[7] = 2.*(q2*q3 + q0*q1);
	rotm[8] = (1. - 2.*(q1*q1 + q2*q2));

	return;
}


__global__ void slicing(float *myslice, int quat_index, int MASKPIX){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// get rotation matrix
	float rotm_cache[9];
	get_rot_mat(quat_index, rotm_cache);

	float q0, q1, q2;
	float d0, d1, d2;
	float inten = 0, ct = 0;

	if(offset >= __pats_gpu[0]*__pats_gpu[1]) return;

	// mask
	x = tex1Dfetch(__tex_mask, offset);
	if(x > MASKPIX){
		myslice[offset] = 0;
		return;
	}

	// detector coordinate
	// transfer to cudaArray coordinate system
	q1 = tex1Dfetch(__tex_det, offset*3);
	q0 = tex1Dfetch(__tex_det, offset*3+1);
	q2 = tex1Dfetch(__tex_det, offset*3+2);
	// d0, d1, d2 are on width, height, depth axis 
	d0 = rotm_cache[0] * q0 + rotm_cache[1] * q1 + rotm_cache[2] * q2;
	d1 = rotm_cache[3] * q0 + rotm_cache[4] * q1 + rotm_cache[5] * q2;
	d2 = rotm_cache[6] * q0 + rotm_cache[7] * q1 + rotm_cache[8] * q2;

	// interp
	int lx, ly, lz, z;
	ct = (__vol_len_gpu[0]-1)/2.0;
	d0 += ct;
	d1 += ct;
	d2 += ct;
	lx = (int) d0;
	ly = (int) d1;
	lz = (int) d2;
	x = lx+1;
	y = ly+1;
	z = lz+1;

	// boundary control
	if(lx<0 || ly<0 || lz<0 || x>=__vol_len_gpu[0] || y>=__vol_len_gpu[0] || z>=__vol_len_gpu[0]){
		myslice[offset] = 0;
		return;
	}

	d0 = d0-lx; d1 = d1-ly; d2 = d2-lz;
	q0 = 1-d0;  q1 = 1-d1;  q2 = 1-d2;

	inten += q0 * q1 * q2 * tex3D(__tex_model, lx, ly, lz);
	inten += d0 * q1 * q2 * tex3D(__tex_model, x, ly, lz);
	inten += d0 * d1 * q2 * tex3D(__tex_model, x, y, lz);
	inten += d0 * d1 * d2 * tex3D(__tex_model, x, y, z);
	inten += d0 * q1 * d2 * tex3D(__tex_model, x, ly, z);
	inten += q0 * d1 * d2 * tex3D(__tex_model, lx, y, z);
	inten += q0 * q1 * d2 * tex3D(__tex_model, lx, ly, z);
	inten += q0 * d1 * q2 * tex3D(__tex_model, lx, y, lz);

	myslice[offset] = inten;

	return;
}



void get_slice(int quat_index, float *myslice, int BlockSize, int det_x, int det_y, int MASKPIX)
{

	gpuInitchk();

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( (det_x+BlockSize-1)/BlockSize, (det_y+BlockSize-1)/BlockSize );

	// slicing
	slicing<<<blocks, threads>>>(__myslice_device, quat_index, MASKPIX);

	// copy back
	if (myslice != NULL)
		cudaErrchk(cudaMemcpy(myslice, __myslice_device, det_x*det_y*sizeof(float), cudaMemcpyDeviceToHost));

}


// anyone who is interested can also optimize it by using shared memory
// I didn't try that cuz its really complex to arrange such a large volume array
// ...
// ok ... cuz I'm lazy ...
__global__ void merging(float *myslice, float *new_model, float *merge_w, int quat_index)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float q0, q1, q2, tmp, val;
	float d0, d1, d2;
	int lx, ly, lz, hx, hy, hz;
	float ct = 0;

	if(offset >= __pats_gpu[0]*__pats_gpu[1]) return;

	// mask
	x = tex1Dfetch(__tex_mask, offset);
	if(x > 1){
		return;
	}

	// quaternion
	float rotm_cache[9];
	get_rot_mat(quat_index, rotm_cache);

	// get detector coordinate
	// and transfer to cudaArray coordinate system (exchange x-axis and y-axis)
	q1 = tex1Dfetch(__tex_det, offset*3);
	q0 = tex1Dfetch(__tex_det, offset*3+1);
	q2 = tex1Dfetch(__tex_det, offset*3+2);
	// coordinate in width axis
	d0 = rotm_cache[0] * q0 + rotm_cache[1] * q1 + rotm_cache[2] * q2;
	// coordinate in height axis
	d1 = rotm_cache[3] * q0 + rotm_cache[4] * q1 + rotm_cache[5] * q2;
	// coordinate in depth axis
	d2 = rotm_cache[6] * q0 + rotm_cache[7] * q1 + rotm_cache[8] * q2;

	ct = (__vol_len_gpu[0]-1)/2.0;
	d0 += ct;
	d1 += ct;
	d2 += ct;
	// index = (?)z * len^2 + (?)y * len + (?)x
	lx = (int) d0;
	ly = (int) d1;
	lz = (int) d2;
	hx = lx+1;
	hy = ly+1;
	hz = lz+1;
	y = __vol_len_gpu[0];
	x = __vol_len_gpu[0] * __vol_len_gpu[0];
	
	// boundary control
	if(lx<0 || ly<0 || lz<0 || hx>=y || hy>=y || hz>=y){
		return;
	}

	// interp
	val = myslice[offset];
	d0 = d0-lx; d1 = d1-ly; d2 = d2-lz;
	q0 = 1-d0;  q1 = 1-d1;  q2 = 1-d2;

	tmp = q0 * q1 * q2;
	atomicAdd(&merge_w[lz * x + ly * y + lx], tmp);
	atomicAdd(&new_model[lz * x + ly * y + lx], tmp*val);

	tmp = q0 * q1 * d2;
	atomicAdd(&merge_w[hz * x + ly * y + lx], tmp);
	atomicAdd(&new_model[hz * x + ly * y + lx], tmp*val);

	tmp = q0 * d1 * d2;
	atomicAdd(&merge_w[hz * x + hy * y + lx], tmp);
	atomicAdd(&new_model[hz * x + hy * y + lx], tmp*val);

	tmp = d0 * d1 * d2;
	atomicAdd(&merge_w[hz * x + hy * y + hx], tmp);
	atomicAdd(&new_model[hz * x + hy * y + hx], tmp*val);

	tmp = d0 * q1 * d2;
	atomicAdd(&merge_w[hz * x + ly * y + hx], tmp);
	atomicAdd(&new_model[hz * x + ly * y + hx], tmp*val);

	tmp = d0 * d1 * q2;
	atomicAdd(&merge_w[lz * x + hy * y + hx], tmp);
	atomicAdd(&new_model[lz * x + hy * y + hx], tmp*val);

	tmp = d0 * q1 * q2;
	atomicAdd(&merge_w[lz * x + ly * y + hx], tmp);
	atomicAdd(&new_model[lz * x + ly * y + hx], tmp*val);

	tmp = q0 * d1 * q2;
	atomicAdd(&merge_w[lz * x + hy * y + lx], tmp);
	atomicAdd(&new_model[lz * x + hy * y + lx], tmp*val);

	return;
}



// this function really fucks off. 8 ms to finish on K80 !
// but do we really need to optimize it ?
// it ONLY runs one time at the end of each reconstruction iteration.
__global__ void merging_sc(float *new_model, float *merge_w, float scale_factor)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float ww;
	long int offset = x + y * blockDim.x * gridDim.x;

	while(offset < __vol_len_gpu[0]*__vol_len_gpu[0]*__vol_len_gpu[0]){

		ww = merge_w[offset];
		if(ww > 0) new_model[offset] = new_model[offset] / ww * scale_factor;

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
	merging<<<blocks, threads>>>(__myslice_device, __model_2_gpu, __w_gpu, quat_index);


}



void merge_scaling(int GridSize, int BlockSize, float scale_factor)
{

	gpuInitchk();

	// check scale factor
	if(scale_factor <= 0) scale_factor = 1;

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( GridSize, GridSize );

	// scaling
	merging_sc<<<blocks, threads>>>(__model_2_gpu, __w_gpu, scale_factor);

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

	// calc, __mapped_array_device_2 is initiated in gpu_var_init()
	angcorr_diff<<<array_length, __ThreadPerBlock>>> (__mapped_array_device_2, partition*partition);
	
	// reduce
	float c = 0;
	c = slice_sum(NULL, 1, array_length, __mapped_array_device_2);
	/*
	int i;
	for(i=0; i<array_length; i++){
		c += __mapped_array_host_2[i];
	}
	*/
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
__global__ void possion_likelihood(float beta, float *reduced_array, float scaling_factor)
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
			w *= scaling_factor;
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



// returned "likelihood" is a float number
double calc_likelihood(float beta, float *model_slice, float *pattern, int det_x, int det_y, float scaling_factor)
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
	
	// calc, __mapped_array_device are initiated in gpu_var_init()
	possion_likelihood<<<array_length, __ThreadPerBlock>>> (beta, __mapped_array_device, scaling_factor);
	
	// reduce
	double c = 0;
	c = slice_sum(NULL, 1, array_length, __mapped_array_device);
	c = exp(c);
	
	// unbind texture
	cudaUnbindTexture(__tex_01);
	cudaUnbindTexture(__tex_02);

	return c;

}


// calculate the first part of log-likelihood
// log(prob_p1) = sum_t { k_dt * log(W_rt) * beta }
__global__ void possion_likelihood_part(int pat_inx, int *dataset_one_loc, int *dataset_mul_loc, int *dataset_mul_val, 
										float *myslice, float beta, float *reduced_array, float scaling_factor)
{
	__shared__ float cache[__ThreadPerBlock];

	int x, one_head, one_len, mul_head, mul_len;
	int inside_pat_loc;
	float correction, k, w, s;
	int global_offset = threadIdx.x + blockIdx.x * blockDim.x;
	int shared_offset = threadIdx.x;

	// this pixel is a one-photon pixel or multi-photon pixel
	one_head = tex1Dfetch(__tex_dataset_pat_head, 4*pat_inx);
	one_len  = tex1Dfetch(__tex_dataset_pat_head, 4*pat_inx+1);
	mul_head = tex1Dfetch(__tex_dataset_pat_head, 4*pat_inx+2);
	mul_len  = tex1Dfetch(__tex_dataset_pat_head, 4*pat_inx+3);

	if(global_offset < one_len){
		// this is an one-photon pixel, k=1;
		inside_pat_loc = dataset_one_loc[one_head + global_offset];
		x = tex1Dfetch(__tex_mask, inside_pat_loc);
		w = myslice[inside_pat_loc];
		if(x > 0 || w <= 0){
			s = 0;
		}
		else{
			correction = tex1Dfetch(__tex_correction, inside_pat_loc);
			s = __logf(scaling_factor * w) * beta / correction;
		}
	}
	else if(global_offset < (one_len+mul_len)){
		// this is a multi-photon pixel
		inside_pat_loc = dataset_mul_loc[mul_head + global_offset - one_len];
		x = tex1Dfetch(__tex_mask, inside_pat_loc);
		w = myslice[inside_pat_loc];
		if(x > 0 || w <= 0){
			s = 0;
		}
		else{
			correction = tex1Dfetch(__tex_correction, inside_pat_loc);
			k = (float) dataset_mul_val[mul_head + global_offset - one_len];
			s = __logf(scaling_factor * w) * k * beta / correction;
		}
	}
	else{
		// out of boundary
		s = 0;
	}

	cache[shared_offset] = s / (__pats_gpu[0]*__pats_gpu[1] - __num_mask_ron_gpu[0] - __num_mask_ron_gpu[1]);

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


// calculate sum_t ( k_dt*log(W_rt)*beta ), the first part of log-likelihood
// photon_len is num_one_photon_pix + num_multi_photon_pixel, it should be <= det_x * det_y
double calc_likelihood_part(int pat_inx, int photon_len, int det_x, int det_y, float beta, float *model_slice, float scaling_factor)
{

	gpuInitchk();

	// device memory
	if (model_slice != NULL)
		cudaErrchk(cudaMemcpy(__myslice_device, model_slice, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));

	int array_length = (photon_len+__ThreadPerBlock-1)/__ThreadPerBlock;

	// calc, __mapped_array_device are initiated in gpu_var_init(), length is (det_x*det_y+__ThreadPerBlock-1)/__ThreadPerBlock;
	// ONLY the first 'array_length' items of __mapped_array_device array are useful information !!!
	possion_likelihood_part<<<array_length, __ThreadPerBlock>>> (pat_inx, __dataset_one_loc, __dataset_mul_loc, 
							__dataset_mul_val, __myslice_device, beta, __mapped_array_device, scaling_factor);

	// sum __mapped_array_device
	double c_p1 = 0;
	c_p1 = slice_sum(NULL, 1, array_length, __mapped_array_device);

	return c_p1;
}


// calculate sum_t ( k_dt*log(W_rt)*beta ), the first part of log-likelihood
// photon_len is num_one_photon_pix + num_multi_photon_pixel, it should be <= det_x * det_y
// stream parallel
double calc_likelihood_part_par(int pat_inx, int photon_len, int det_x, int det_y, float beta, 
									float *model_slice, float scaling_factor, int thread_id)
{

	gpuInitchk();

	// device memory
	if (model_slice != NULL)
		cudaErrchk(cudaMemcpy(__myslice_device, model_slice, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));

	int array_length = (photon_len+__ThreadPerBlock-1)/__ThreadPerBlock;
	int offset = (det_x*det_y+__ThreadPerBlock-1)/__ThreadPerBlock;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
printf("\nprocessed %d,", pat_inx);
	// calc, __mapped_array_device are initiated in gpu_var_init(), length is (det_x*det_y+__ThreadPerBlock-1)/__ThreadPerBlock;
	// ONLY the first 'array_length' items of __mapped_array_device array are useful information !!!
	possion_likelihood_part<<<array_length, __ThreadPerBlock, 0, stream>>> (pat_inx, __dataset_one_loc, __dataset_mul_loc, 
				__dataset_mul_val, __myslice_device, beta, __mapped_array_device+offset*thread_id, scaling_factor);
printf("then,");
	// sum __mapped_array_device
	double c_p1 = 0;
	c_p1 = slice_sum_par(NULL, 1, array_length, __mapped_array_device+offset*thread_id, stream);
printf("done\n");
	cudaStreamDestroy(stream);
	return c_p1;

}



// __tex_mask should be initiated in advance
// pattern should be given if add == 1, otherwise set pattern=NULL
__global__ void pat_dot_float(float number, float *new_slice, float *pattern, int add)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float k;

	x = tex1Dfetch(__tex_mask, offset);
	if(offset >= __pats_gpu[0]*__pats_gpu[1] || x > 1) return;

	if(add){
		//k = tex1Dfetch(__tex_01, offset);
		k = pattern[offset];
		new_slice[offset] += k * number;
	}
	else{
		new_slice[offset] *= number;
	}

}

// __tex_mask should be initiated in advance
// maximization dot, only process one photon pixels
__global__ void max_dot_partone(int pat_inx, float prob, int *one_loc, float *new_slice)
{
	
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	int x, one_head, one_len, loc;

	// this pixel is a one-photon pixel or multi-photon pixel
	one_head = tex1Dfetch(__tex_dataset_pat_head, 4*pat_inx);
	one_len  = tex1Dfetch(__tex_dataset_pat_head, 4*pat_inx+1);

	if(offset >= one_len) return;
	loc = one_loc[one_head + offset];

	// mask and boundary control
	x = tex1Dfetch(__tex_mask, loc);
	if(loc >= __pats_gpu[0]*__pats_gpu[1] || x > 1) return;

	// dot
	new_slice[loc] += prob;

}

// __tex_mask should be initiated in advance
// maximization dot, only process multiple photon pixels
__global__ void max_dot_partmul(int pat_inx, float prob, int *mul_loc, int *mul_val, float *new_slice)
{
	
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	int x, mul_head, mul_len, loc, val;

	// this pixel is a one-photon pixel or multi-photon pixel
	mul_head = tex1Dfetch(__tex_dataset_pat_head, 4*pat_inx+2);
	mul_len  = tex1Dfetch(__tex_dataset_pat_head, 4*pat_inx+3);

	if(offset >= mul_len) return;
	loc = mul_loc[mul_head + offset];
	val = mul_val[mul_head + offset];

	// mask and boundary control
	x = tex1Dfetch(__tex_mask, loc);
	if(loc >= __pats_gpu[0]*__pats_gpu[1] || x > 1) return;

	// dot
	new_slice[loc] += prob * val;

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

	//cudaErrchk(cudaBindTexture(NULL, __tex_01, __mypattern_device, det_x*det_y*sizeof(float)));

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( (det_x+BlockSize-1)/BlockSize, (det_y+BlockSize-1)/BlockSize );

	// calc
	pat_dot_float<<<blocks, threads>>>(prob, __myslice_device, __mypattern_device, 1);

	// unbind texture
	//cudaUnbindTexture(__tex_01);

	// copy back ?
	if(new_slice != NULL)
		download_currSlice_from_gpu(new_slice, det_x, det_y);

}

// DO NOT FORGET to initiate __myslice_device to 0 in advance !!!
void maximization_dot_part(int pat_inx, float prob, int one_pix_count, int mul_pix_count)
{
	gpuInitchk();

	//cudaErrchk(cudaBindTexture(NULL, __tex_01, __mypattern_device, det_x*det_y*sizeof(float)));
	int array_length;

	// calc one photon pix
	array_length = (one_pix_count+__ThreadPerBlock-1)/__ThreadPerBlock;
	max_dot_partone<<<array_length, __ThreadPerBlock>>>(pat_inx, prob, __dataset_one_loc, __myslice_device);

	// calc multiple photon pix
	array_length = (mul_pix_count+__ThreadPerBlock-1)/__ThreadPerBlock;
	max_dot_partmul<<<array_length, __ThreadPerBlock>>>(pat_inx, prob, __dataset_mul_loc, __dataset_mul_val, __myslice_device);

	// unbind texture
	//cudaUnbindTexture(__tex_01);

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
	pat_dot_float<<<blocks, threads>>>(scaling_factor, __myslice_device, NULL, 0);

}


/*
mode 1 : arr_1 = arr_1 * arr_2
mode 2 : arr_1 = arr_1 / arr_2
mode 3 : arr_1 = arr_2 / arr_1
*/
__global__ void arr_dot_arr(float *arr_1, float *arr_2, int length, int mode)
{
	
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	float k, m;
	k = arr_1[offset];
	m = arr_2[offset];

	if(offset >= length) return;

	switch (mode){
		case 1:
			arr_1[offset] = k * m;
			break;
		case 2:
			arr_1[offset] = k / m;
			break;
		case 3:
			arr_1[offset] = m / k;
			break;
	}

}

void update_scaling(int num_data)
{
// TODO : check correction
	gpuInitchk();

	int gridsize = (num_data + __ThreadPerBlock -1) / __ThreadPerBlock;
	// calc
	arr_dot_arr<<<gridsize, __ThreadPerBlock>>>(__scaling_factor_gpu, __photon_count, num_data, 3);

}
