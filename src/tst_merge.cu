#include "base.h"
#include "base_cuda.h"


uint32 __qmax_len;
uint32 __det_x, __det_y;


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


__global__ void slicing(float *ori_det, float *myslice){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float q0, q1, q2;
	float d[3];
	float inten = 0, w = 0;

	if(offset >= __pats_gpu[0]*__pats_gpu[1]) return;
	x = tex1Dfetch(__tex_mask, offset);
	if(x > 0){
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

	if(lx>=0 && ly>=0 && lz>=0){
		q0 = d[0]-lx;
		q1 = d[1]-ly;
		q2 = d[2]-lz;
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx, ly, lz);
	}

	if(lx+1<__vol_len_gpu[0] && ly>=0 && lz>=0){
		q0 = lx+1-d[0];
		q1 = d[1]-ly;
		q2 = d[2]-lz;
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx+1, ly, lz);
	}

	if(lx+1<__vol_len_gpu[0] && ly+1<__vol_len_gpu[0] && lz>=0){
		q0 = lx+1-d[0];
		q1 = ly+1-d[1];
		q2 = d[2]-lz;
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx+1, ly+1, lz);
	}

	if(lx+1<__vol_len_gpu[0] && ly+1<__vol_len_gpu[0] && lz+1<__vol_len_gpu[0]){
		q0 = lx+1-d[0];
		q1 = ly+1-d[1];
		q2 = lz+1-d[2];
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx+1, ly+1, lz+1);
	}

	if(lx+1<__vol_len_gpu[0] && ly>=0 && lz+1<__vol_len_gpu[0]){
		q0 = lx+1-d[0];
		q1 = d[1]-ly;
		q2 = lz+1-d[2];
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx+1, ly, lz+1);
	}

	if(lx>=0 && ly+1<__vol_len_gpu[0] && lz+1<__vol_len_gpu[0]){
		q0 = d[0]-lx;
		q1 = ly+1-d[1];
		q2 = lz+1-d[2];
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx, ly+1, lz+1);
	}

	if(lx>=0 && ly>=0 && lz+1<__vol_len_gpu[0]){
		q0 = d[0]-lx;
		q1 = d[1]-ly;
		q2 = lz+1-d[2];
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx, ly, lz+1);
	}

	if(lx>=0 && ly+1<__vol_len_gpu[0] && lz>=0){
		q0 = d[0]-lx;
		q1 = ly+1-d[1];
		q2 = d[2]-lz;
		q2 = sqrt( q0*q0 + q1*q1 + q2*q2 );
		q2 = __expf(-q2/0.3f);
		w += q2;
		inten += q2 * tex3D(__tex_model, lx, ly+1, lz);
	}

	if(w<1e-6)
		myslice[offset] = 0;// / w;
	else
		myslice[offset] = inten / w;

	//printf("test_id=%d, inten=%f, w=%f\n", offset, inten, w);

	return;
}



void get_slice(float *quaternion, float *ori_det, float *myslice, int BlockSize){

	float *myslice_device;
	cudaErrchk(cudaMalloc((void**)&myslice_device, __det_x*__det_y*sizeof(float)));
	// copy det
	cudaErrchk(cudaMemcpy(__det_gpu, ori_det, __det_x*__det_y*sizeof(float)*3, cudaMemcpyHostToDevice));

	// rotation matrix
	float rotm[9];
	get_rotate_matrix(quaternion, rotm);
	cudaErrchk(cudaMemcpyToSymbol(__rotm_gpu, rotm, sizeof(float)*9));

	// dim
	dim3 threads( BlockSize, BlockSize );
	dim3 blocks( (__det_x+BlockSize-1)/BlockSize, (__det_y+BlockSize-1)/BlockSize );

	// slicing
	slicing<<<blocks, threads>>>(__det_gpu, myslice_device);

	// copy back
	cudaMemcpy(myslice, myslice_device, __det_x*__det_y*sizeof(float), cudaMemcpyDeviceToHost);

	// cudafree
	cudaFree(myslice_device);

}






int main(int argc, char** argv){
	char fn[999], mask_fn[999], det_fn[999];
	int c, gpu;
	cudaError_t custatus;
	int BlockSize = 16;
	gpu = 0;

	float *det, *mymodel, *myslice;
	
	int *mask;

	float quater[] = {0.0168, 0.8026, 0.5956, 0.0292};

	int pat_s[2];

	FILE* fp;

	while( (c = getopt(argc, argv, "f:d:x:y:l:m:g:h")) != -1 ){
		switch(c){
			case 'f':
				strcpy(fn, optarg);
				break;
			case 'd':
				strcpy(det_fn, optarg);
				break;
			case 'x':
				__det_x = (uint32)atoi(optarg);
				break;
			case 'y':
				__det_y = (uint32)atoi(optarg);
				break;
			case 'l':
				__qmax_len = (uint32)atoi(optarg);
				break;
			case 'm':
				strcpy(mask_fn, optarg);
				break;
			case 'g':
				gpu = atoi(optarg);
				break;
			case 'h':
				printf("options:\n");
				printf("         -f [signal file]\n");
				printf("         -d [detector file]\n");
				printf("         -x [pattern pat_s[0]]\n");
				printf("         -y [pattern pat_s[1]]\n");
				printf("         -l [length of volume]\n");
				printf("         -m [mask file]\n");
				printf("         -g [GPU number]\n");
				printf("         -h help\n");
				return 0;
			default:
				printf("Do nothing.\n");
				break;
		}
	}

	custatus = cudaSetDevice(gpu);
    if(custatus != cudaSuccess){
        printf("Failed to set Device %d. Exit\n", gpu);
        return -1;
    }

    pat_s[0] = (int)__det_x;
    pat_s[1] = (int)__det_y;


    // init global gpu var
    cudaMalloc((void**)&__det_gpu, pat_s[0]*pat_s[1]*sizeof(float)*3);
    cudaMalloc((void**)&__mask_gpu, pat_s[0]*pat_s[1]*sizeof(int));
    cudaChannelFormatDesc volDesc = cudaCreateChannelDesc<float>();
	cudaExtent volExt = make_cudaExtent((int)__qmax_len, (int)__qmax_len, (int)__qmax_len);
	cudaMalloc3DArray(&__model_1_gpu, &volDesc, volExt);

	// init gpu constant
	cudaMemcpyToSymbol(__pats_gpu, pat_s, sizeof(int)*2);
	int tmp[] = {(int)__qmax_len};
	cudaMemcpyToSymbol(__vol_len_gpu, tmp, sizeof(int));


    // read mask file
    mask = (int*) malloc(pat_s[0]*pat_s[1]*sizeof(int));
    fp = fopen(mask_fn, "rb");
    if( fp == NULL ){
		printf("[error] Your mask file is invalid.\n");
		return -1;
	}
    fread(mask, sizeof(int), pat_s[0]*pat_s[1], fp);
    fclose(fp);
    
    cudaMemcpy(__mask_gpu, mask, pat_s[0]*pat_s[1]*sizeof(int), cudaMemcpyHostToDevice);
    cudaErrchk(cudaBindTexture(NULL, __tex_mask, __mask_gpu, pat_s[0]*pat_s[1]*sizeof(int)));


    // read det file
    det = (float*) malloc(pat_s[0]*pat_s[1]*sizeof(float)*3);
	fp = fopen(det_fn, "r");
	if(fp == NULL){
		printf("[ERROR] Your detector file is invalid.\n");
		return false;
	}
	char line[999];
	int line_num = 0;

	while (fgets(line, 999, fp) != NULL) {
		if(line_num < pat_s[0] * pat_s[1]){
			det[line_num*3] = atof(strtok(line, " \n"));
			det[line_num*3+1] = atof(strtok(NULL, " \n"));
			det[line_num*3+2] = atof(strtok(NULL, " \n"));
			mask[line_num] = atoi(strtok(NULL, " \n"));
			line_num ++;
		}
	}

	fclose(fp);

    // __tex_model
    mymodel = (float*) malloc(__qmax_len*__qmax_len*__qmax_len*sizeof(float));
    fp = fopen(fn, "rb");
    if( fp == NULL ){
		printf("[error] Your data file is invalid.\n");
		return -1;
	}
	fread(mymodel, sizeof(float), __qmax_len*__qmax_len*__qmax_len, fp);
	fclose(fp);

	cudaMemcpy3DParms volParms = {0};
	volParms.srcPtr = make_cudaPitchedPtr((void*)mymodel, sizeof(float)*__qmax_len, (int)__qmax_len, (int)__qmax_len);
	volParms.dstArray = __model_1_gpu;
	volParms.extent = volExt;
	volParms.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&volParms);
	cudaBindTextureToArray(__tex_model, __model_1_gpu);

	// check error
	cudaErrchk(cudaDeviceSynchronize());


	// test performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);



	// slicing
	myslice = (float*) malloc(__det_x*__det_y*sizeof(float));
	get_slice(quater, det, myslice, BlockSize);
	
	


	// test performance
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float estime;
	cudaEventElapsedTime(&estime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("use %.5f ms\n", estime);


	// write
	fp = fopen("tst_merge.bin", "wb");
	fwrite(myslice, sizeof(float), pat_s[0]*pat_s[1], fp);
	fclose(fp);


	// free malloc
	cudaUnbindTexture(__tex_model);
	cudaFreeArray(__model_1_gpu);
	cudaUnbindTexture(__tex_mask);
	cudaFree(__mask_gpu);
	cudaFree(__det_gpu);
	cudaDeviceReset();
	free(mymodel);
	free(det);
	free(mask);
	free(myslice);

}