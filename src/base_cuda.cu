#include "base_cuda.h"



/* All functions for C modules to call  */

extern "C" void setDevice(int gpu_id);

extern "C" void gpu_var_init(int det_x, int det_y, float det_center[2], int vol_size, int stoprad, 
	float *ori_det, int *ori_mask, float *init_model_1, float *init_model_2, float *init_merge_w);

extern "C" void download_model2_from_gpu(float *model_2, int vol_size);

extern "C" void free_cuda_all();

extern "C" void cuda_start_event();

extern "C" float cuda_return_time();

extern "C" void get_slice(float *quaternion, float *myslice, int BlockSize, int det_x, int det_y, int MASKPIX);

extern "C" void merge_slice(float *quaternion, float *myslice, int BlockSize, int det_x, int det_y);

extern "C" void merge_scaling(int GridSize, int BlockSize);


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



void gpu_var_init(int det_x, int det_y, float det_center[2], int vol_size, int stoprad, 
	float *ori_det, int *ori_mask, float *init_model_1, float *init_model_2, float *init_merge_w)
{
	// constant
	int pat_s[] = {det_x, det_y};
	int vols[] = {vol_size};
	int stopr[] = {stoprad};
	cudaMemcpyToSymbol(__vol_len_gpu, vols, sizeof(int));
	cudaMemcpyToSymbol(__pats_gpu, pat_s, sizeof(int)*2);
	cudaMemcpyToSymbol(__center_gpu, det_center, sizeof(float)*2);
	cudaMemcpyToSymbol(__stoprad_gpu, stopr, sizeof(int));


	// mask & det
	cudaMalloc((void**)&__det_gpu, pat_s[0]*pat_s[1]*sizeof(float)*3);
	cudaMemcpy(__det_gpu, ori_det, pat_s[0]*pat_s[1]*sizeof(float)*3, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&__mask_gpu, pat_s[0]*pat_s[1]*sizeof(int));
	cudaMemcpy(__mask_gpu, ori_mask, pat_s[0]*pat_s[1]*sizeof(int), cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, __tex_mask, __mask_gpu, pat_s[0]*pat_s[1]*sizeof(int));


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


	// check error
	cudaErrchk(cudaDeviceSynchronize());

	// change init flag
	__initiated = 1;
}



void download_model2_from_gpu(float *new_model_2, int vol_size)
{
	cudaMemcpy(new_model_2, __model_2_gpu, 
		vol_size*vol_size*vol_size*sizeof(float), cudaMemcpyDeviceToHost);
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

	// free model_1
	cudaUnbindTexture(__tex_model);
	cudaFreeArray(__model_1_gpu);

	// free model_2
	cudaFree(__model_2_gpu);

	// free merge_w
	cudaFree(__w_gpu);

	// destroy cuda event
	cudaEventDestroy(__custart);
	cudaEventDestroy(__custop);

	// reset
	cudaDeviceReset();

	// change init flag
	__initiated = 0;
}



void cuda_start_event()
{
	cudaErrchk(cudaEventCreate(&__custart));
    cudaErrchk(cudaEventCreate(&__custop));
    cudaErrchk(cudaEventRecord(__custart, 0));
}



float cuda_return_time()
{
	float estime;
	cudaErrchk(cudaEventRecord(__custop, 0));
	cudaErrchk(cudaEventSynchronize(__custop));
	cudaErrchk(cudaEventElapsedTime(&estime, __custart, __custop));
	cudaEventDestroy(__custart);
	cudaEventDestroy(__custop);
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


__global__ void slicing(float *ori_det, float *myslice, int *mymask, int MASKPIX){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

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
	d[0] = __rotm_gpu[0] * q0 + __rotm_gpu[1] * q1 + __rotm_gpu[2] * q2;
	d[1] = __rotm_gpu[3] * q0 + __rotm_gpu[4] * q1 + __rotm_gpu[5] * q2;
	d[2] = __rotm_gpu[6] * q0 + __rotm_gpu[7] * q1 + __rotm_gpu[8] * q2;

	// interp
	int lx, ly, lz, center;
	center = (__vol_len_gpu[0]-1)/2;
	d[0] = d[0] + center;
	d[1] = d[1] + center;
	d[2] = d[2] + center;
	lx = (int) floor(d[0]);
	ly = (int) floor(d[1]);
	lz = (int) floor(d[2]);

	if(d[0]>=__vol_len_gpu[0] || d[1]>=__vol_len_gpu[0] || d[2]>=__vol_len_gpu[0] || d[0]<0 || d[1]<0 || d[2]<0){
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



void get_slice(float *quaternion, float *myslice, int BlockSize, int det_x, int det_y, int MASKPIX)
{

	gpuInitchk();

	float *myslice_device;
	cudaErrchk(cudaMalloc((void**)&myslice_device, det_x*det_y*sizeof(float)));

	// rotation matrix
	float rotm[9];
	get_rotate_matrix(quaternion, rotm);
	cudaErrchk(cudaMemcpyToSymbol(__rotm_gpu, rotm, sizeof(float)*9));

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( (det_x+BlockSize-1)/BlockSize, (det_y+BlockSize-1)/BlockSize );

	// slicing
	slicing<<<blocks, threads>>>(__det_gpu, myslice_device, __mask_gpu, MASKPIX);

	// copy back
	cudaErrchk(cudaMemcpy(myslice, myslice_device, det_x*det_y*sizeof(float), cudaMemcpyDeviceToHost));

	// cudafree
	cudaErrchk(cudaFree(myslice_device));

}



// anyone who is interested can also optimized it using shared memory
// I didn't try that cuz its really complex to arrange such a large volume array
// ...
// ok ... cuz I'm lazy ...
__global__ void merging(float *ori_det, float *myslice, int *mymask, float *new_model, float *merge_w)
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

	// calculate rotated det coordinate
	q0 = ori_det[offset*3];
	q1 = ori_det[offset*3+1];
	q2 = ori_det[offset*3+3];
	d[0] = __rotm_gpu[0] * q0 + __rotm_gpu[1] * q1 + __rotm_gpu[2] * q2;
	d[1] = __rotm_gpu[3] * q0 + __rotm_gpu[4] * q1 + __rotm_gpu[5] * q2;
	d[2] = __rotm_gpu[6] * q0 + __rotm_gpu[7] * q1 + __rotm_gpu[8] * q2;

	center = (__vol_len_gpu[0]-1)/2;
	d[0] = d[0] + center;
	d[1] = d[1] + center;
	d[2] = d[2] + center;
	lx = (long int) floor(d[0]);
	ly = (long int) floor(d[1]);
	lz = (long int) floor(d[2]);

	val = myslice[offset];

	// boundary control
	if(d[0]>=__vol_len_gpu[0] || d[1]>=__vol_len_gpu[0] || d[2]>=__vol_len_gpu[0] || d[0]<0 || d[1]<0 || d[2]<0){
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
__global__ void merging_sc(float *new_model, float *merge_w)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	long int offset = x + y * blockDim.x * gridDim.x;

	while(offset < __vol_len_gpu[0]*__vol_len_gpu[0]*__vol_len_gpu[0]){
		
		new_model[offset] = new_model[offset] / merge_w[offset];

		offset += blockDim.x * gridDim.x * blockDim.y * gridDim.y;
	}

}



void merge_slice(float *quaternion, float *myslice, int BlockSize, int det_x, int det_y)
{

	gpuInitchk();

	float *myslice_device;
	cudaErrchk(cudaMalloc((void**)&myslice_device, det_x*det_y*sizeof(float)));
	cudaErrchk(cudaMemcpy(myslice_device, myslice, det_x*det_y*sizeof(float), cudaMemcpyHostToDevice));

	// rotation matrix
	float rotm[9];
	get_rotate_matrix(quaternion, rotm);
	cudaErrchk(cudaMemcpyToSymbol(__rotm_gpu, rotm, sizeof(float)*9));

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( (det_x+BlockSize-1)/BlockSize, (det_y+BlockSize-1)/BlockSize );

	// merging
	merging<<<blocks, threads>>>(__det_gpu, myslice_device, __mask_gpu, __model_2_gpu, __w_gpu);

	// cudafree
	cudaErrchk(cudaFree(myslice_device));

}



void merge_scaling(int GridSize, int BlockSize)
{

	gpuInitchk();

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( GridSize, GridSize );

	// scaling
	merging_sc<<<blocks, threads>>>(__model_2_gpu, __w_gpu);

}


/*********************************************/

/*            Angular Correlation            */

/*********************************************/
