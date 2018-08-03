#include "base.h"
#include "base_cuda.h"

//int pat_s[2];
//int stoprad;
int __stoprad[1];
uint32 __det_x, __det_y;


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


__global__ void cu_complex_l(cufftComplex* mycomplex, cufftReal* rets, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<N){
		rets[i] = mycomplex[i].x * mycomplex[i].x + mycomplex[i].y * mycomplex[i].y;
	}
	return;
}

void do_angcorr(int partition, cufftReal* pat, cufftReal *result, int BlockSize){

	// locate device memory
	cufftReal *pat_device;
	cudaMalloc((void**)&pat_device, __det_x*__det_y*sizeof(cufftReal));
	cudaErrchk(cudaBindTexture(NULL, __tex_01, pat_device, __det_x*__det_y*sizeof(cufftReal)))

	cufftReal *polar_device;
	cudaMalloc((void**)&polar_device, partition*partition*sizeof(cufftReal));

	cufftComplex *re_device;
	cudaMalloc((void**)&re_device, (partition/2+1)*partition*sizeof(cufftComplex));

	cufftReal *norm_device;
	cudaMalloc((void**)&norm_device, (partition/2+1)*partition*sizeof(cufftReal));

	// memcpy
	cudaMemcpy(pat_device, pat, __det_x*__det_y*sizeof(cufftReal), cudaMemcpyHostToDevice);

	// polar grid
	dim3 threads(BlockSize,BlockSize);
	dim3 blocks((partition+BlockSize-1)/BlockSize, (partition+BlockSize-1)/BlockSize);
	polar_transfer<<<blocks, threads>>>(polar_device, partition);

	// init cufft handler
	cufftHandle plan_many_signal;
	cufftPlan1d(&plan_many_signal, partition, CUFFT_R2C, partition);

	// run fft
	cufftErrchk(cufftExecR2C(plan_many_signal, polar_device, re_device));

	// self dot
	cu_complex_l<<<partition,(partition/2+1)>>>(re_device, norm_device, (partition/2+1)*partition);

	// memcpy
	cudaMemcpy(result, norm_device, (partition/2+1)*partition*sizeof(cufftReal), cudaMemcpyDeviceToHost);

	// destroy
	cufftDestroy(plan_many_signal);
	cudaUnbindTexture(__tex_01);
	cudaFree(pat_device);
	cudaFree(re_device);
	cudaFree(norm_device);
	cudaFree(polar_device);

}


int main(int argc, char** argv){
	char fn[999], mask_fn[999];
	int c, gpu;
	int bins;
	cudaError_t custatus;
	gpu = 0;
	__stoprad[0]= 20;

	while( (c = getopt(argc, argv, "f:x:y:p:m:g:t:h")) != -1 ){
		switch(c){
			case 'f':
				strcpy(fn, optarg);
				break;
			case 'x':
				__det_x = (uint32)atoi(optarg);
				break;
			case 'y':
				__det_y = (uint32)atoi(optarg);
				break;
			case 'p':
				bins = atoi(optarg);
				break;
			case 'm':
				strcpy(mask_fn, optarg);
				break;
			case 'g':
				gpu = atoi(optarg);
				break;
			case 't':
				__stoprad[0] = atoi(optarg);
				break;
			case 'h':
				printf("options:\n");
				printf("         -f [signal file]\n");
				printf("         -x [pattern pat_s[0]]\n");
				printf("         -y [pattern pat_s[1]]\n");
				printf("         -p [number of partition bins]\n");
				printf("         -g [GPU number]\n");
				printf("         -t [stop radius]\n");
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

    int pat_s[] = {(int)__det_x, (int)__det_y};

    // test performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	// locate host memory
	cufftReal* signalx = (float*) malloc(pat_s[0]*pat_s[1]*sizeof(cufftReal));
	int *mask = (int*) malloc(pat_s[0]*pat_s[1]*sizeof(int));
	cufftReal* result = (float*) malloc(bins*(bins/2+1)*sizeof(float));
	float center[2];
	center[0] = (pat_s[0]-1)/2.0;
	center[1] = (pat_s[1]-1)/2.0;

	// size
	cudaMemcpyToSymbol(__pats_gpu, pat_s, sizeof(int)*2);
	cudaMemcpyToSymbol(__center_gpu, center, sizeof(float)*2);
	cudaMemcpyToSymbol(__stoprad_gpu, __stoprad, sizeof(int));

	// init sig
	FILE *fp;
	fp = fopen(fn, "rb");
	if(fp == NULL){
		printf("Error. Do not load file\n");
		return -1;
	}
	fread(signalx, sizeof(cufftReal), pat_s[1]*pat_s[0], fp);
	fclose(fp);

	// mask
	fp = fopen(mask_fn, "rb");
	if(fp == NULL){
		printf("Error. Do not load mask\n");
		return -1;
	}
	fread(mask, sizeof(int), pat_s[1]*pat_s[0], fp);
	fclose(fp);
	cudaMalloc((void**)&__mask_gpu, pat_s[0]*pat_s[1]*sizeof(int));
	cudaMemcpy(__mask_gpu, mask, pat_s[0]*pat_s[1]*sizeof(int), cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, __tex_mask, __mask_gpu, pat_s[0]*pat_s[1]*sizeof(int));

	// fft
	do_angcorr(bins, signalx, result, 16);

	// test performance
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float estime;
	cudaEventElapsedTime(&estime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("use %.5f ms\n", estime);



	//save
	fp = fopen("test_cufft.bin", "wb");
	fwrite(result, sizeof(cufftReal), bins*(bins/2+1), fp);
	fclose(fp);
	// free
	free(signalx);
	free(result);
	free(mask);
	cudaUnbindTexture(__tex_mask);
	cudaFree(__mask_gpu);
	cudaDeviceReset();

}