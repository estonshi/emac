#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

#include "base.h"


int do_cufft(int N_signal, int L_signal, float* pat, cufftComplex *result, int BlockSize){
	// locate device memory
	float *pat_device;
	cudaMalloc((void**)&pat_device, L_signal*N_signal*sizeof(float));
	cufftComplex *re_device;
	cudaMalloc((void**)&re_device, L_signal*N_signal*sizeof(cufftComplex));

	// memcpy
	cudaMemcpy(pat_device, pat, L_signal*N_signal*sizeof(float), cudaMemcpyHostToDevice);

	// init block & grid
	dim3 myblock2D( BlockSize, BlockSize );
	dim3 mygrid2D( (L_signal+BlockSize-1)/myblock2D.x , (N_signal+BlockSize-1)/myblock2D.y );

	// init cufft handler
	cufftHandle plan_many_signal;
	int number_n[1] = {L_signal};
	cufftPlanMany(&plan_many_signal, 1, number_n, NULL, 1, L_signal, NULL, 1, L_signal, CUFFT_R2C, N_signal);

	// run fft
	cufftExecR2C(plan_many_signal, pat_device, re_device);

	// memcpy
	cudaMemcpy(result, re_device, L_signal*N_signal*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	// destroy
	cufftDestroy(plan_many_signal);
	cudaFree(pat_device);
	cudaFree(re_device);

	return 1;
}


int main(int argc, char** argv){
	char test_item[999];
	int c;
	int sigL, sigN;

	while( (c = getopt(argc, argv, "i:n:l:h")) != -1 ){
		switch(c){
			case 'i':
				strcpy(test_item, optarg);
				break;
			case 'n':
				sigN = atoi(optarg);
				break;
			case 'l':
				sigL = atoi(optarg);
				break;
			case 'h':
				printf("options: -i [test program] , 'cufft' or 'pol_coor' or 'ang_corr'\n");
				printf("         -n [number of signals], for 'cufft'\n");
				printf("         -l [length of signals], for 'cufft'\n");
				printf("         -h help\n");
				break;
			default:
				printf("Do nothing.\n");
				break;
		}
	}

	if(strcmp(test_item, "cufft") == 0){
		// locate host memory
		float* signalx = (float*) malloc(sigL*sigN*sizeof(float));
		cufftComplex* result = (cufftComplex*) malloc(sigN*sigL*sizeof(cufftComplex));
		float* signaly = (float*) malloc(sigL*sigN*sizeof(float));
		// init sig
		for(int i=0; i<sigN*sigL; i++){
			signalx[i] = float((rand() * rand()) % sigL) / sigL;
		}
		// fft
		do_cufft(sigN, sigL, signalx, result, 64);
		for(int i=0; i<sigN*sigL; i++){
			signaly[i] = (float)(result[i].x * result[i].x + result[i].y * result[i].y);
		}
		//save
		FILE *fp;
		fp = fopen("test_cufft.bin", "wb");
		fwrite(signaly, sizeof(float), sigN*sigL, fp);
		fclose(fp);
		// free
		free(signalx);
		free(result);
		free(signaly);
	}
}