#include "base.h"
#include "cuda_funcs.h"

//int pat_s[2];
//int stoprad;
int __stoprad;
uint32 __det_x, __det_y;


int main(int argc, char** argv){
	char fn[999], mask_fn[999];
	int c, gpu;
	int bins;
	gpu = 0;
	__stoprad= 20;
	int __qmax_len = 128;

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
				__stoprad = atoi(optarg);
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

	setDevice(gpu);

	int pat_s[] = {(int)__det_x, (int)__det_y};


	// locate host memory
	float* signalx = (float*) malloc(pat_s[0]*pat_s[1]*sizeof(float));
	int *mask = (int*) malloc(pat_s[0]*pat_s[1]*sizeof(int));
	float* result = (float*) malloc(bins*(bins/2+1)*sizeof(float));
	float center[2];
	center[0] = (pat_s[0]-1)/2.0;
	center[1] = (pat_s[1]-1)/2.0;


	// init sig
	FILE *fp;
	fp = fopen(fn, "rb");
	if(fp == NULL){
		printf("Error. Do not load file\n");
		return -1;
	}
	fread(signalx, sizeof(float), pat_s[1]*pat_s[0], fp);
	fclose(fp);

	// mask
	fp = fopen(mask_fn, "rb");
	if(fp == NULL){
		printf("Error. Do not load mask\n");
		return -1;
	}
	fread(mask, sizeof(int), pat_s[1]*pat_s[0], fp);
	fclose(fp);


	// fake det, mymodel, model_2 and merge_w. All set 0
	float *det, *mymodel, *model_2, *merge_w;
	det = (float*) calloc(pat_s[0]*pat_s[1]*3, sizeof(float));
	mymodel = (float*) calloc(__qmax_len*__qmax_len*__qmax_len, sizeof(float));
	model_2 = (float*) calloc(__qmax_len*__qmax_len*__qmax_len, sizeof(float));
	merge_w = (float*) calloc(__qmax_len*__qmax_len*__qmax_len, sizeof(float));


	// init gpu var
	gpu_var_init((int)__det_x, (int)__det_y, center, __qmax_len, (int)__stoprad, 
										det, mask, mymodel, model_2, merge_w);


	// test performance
	cuda_start_event();

	// fft
	do_angcorr(bins, signalx, result, pat_s[0], pat_s[1], 16);

	// test performance
	float estime = cuda_return_time();
	printf("use %.5f ms\n", estime);


	//save
	fp = fopen("test_cufft.bin", "wb");
	fwrite(result, sizeof(float), bins*(bins/2+1), fp);
	fclose(fp);
	// free
	free(signalx);
	free(result);
	free(mask);
	free(det);
	free(mymodel);
	free(model_2);
	free(merge_w);
	free_cuda_all();

}