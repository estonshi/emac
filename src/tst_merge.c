#include "base.h"


extern void setDevice(int gpu_id);

extern void gpu_var_init(int det_x, int det_y, float det_center[2], int vol_size, int stoprad, 
	float *ori_det, int *ori_mask, float *init_model_1, float *init_model_2, float *init_merge_w);

extern void free_cuda_all();

extern void get_slice(float *quaternion, float *myslice, int BlockSize, int det_x, int det_y, int MASKPIX);

extern void cuda_start_event();

extern float cuda_return_time();
 

uint32 __qmax_len;
uint32 __det_x, __det_y;
 


int main(int argc, char** argv){
	char fn[999], mask_fn[999], det_fn[999];
	int c, gpu;
	int BlockSize = 16;
	gpu = 0;

	float *det, *mymodel, *model_2, *merge_w, *myslice;
	
	int *mask;

	float quater[] = {0.0168, 0.8026, 0.5956, 0.0292};

	int pat_s[2];
	float center[2];

	int i;

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

	setDevice(gpu);

    pat_s[0] = (int)__det_x;
    pat_s[1] = (int)__det_y;

	center[0] = (pat_s[0]-1)/2.0;
	center[1] = (pat_s[1]-1)/2.0;


    // read mask file
    mask = (int*) malloc(pat_s[0]*pat_s[1]*sizeof(int));
    fp = fopen(mask_fn, "rb");
    if( fp == NULL ){
		printf("[error] Your mask file is invalid.\n");
		return -1;
	}
    fread(mask, sizeof(int), pat_s[0]*pat_s[1], fp);
    fclose(fp);
    

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


    // model_1
    mymodel = (float*) malloc(__qmax_len*__qmax_len*__qmax_len*sizeof(float));
    fp = fopen(fn, "rb");
    if( fp == NULL ){
		printf("[error] Your data file is invalid.\n");
		return -1;
	}
	fread(mymodel, sizeof(float), __qmax_len*__qmax_len*__qmax_len, fp);
	fclose(fp);

	// model_2
	model_2 = (float*) calloc(__qmax_len*__qmax_len*__qmax_len, sizeof(float));

	// merge_w
	merge_w = (float*) malloc(__qmax_len*__qmax_len*__qmax_len*sizeof(float));
	for(i=0; i<__qmax_len*__qmax_len*__qmax_len; i++){
		merge_w[i] = 1.0f;
	}


	// init gpu var
	gpu_var_init((int)__det_x, (int)__det_y, center, (int)__qmax_len, 20, 
										det, mask, mymodel, model_2, merge_w);


	// test performance
    cuda_start_event();
    



	// slicing
	myslice = (float*) malloc(__det_x*__det_y*sizeof(float));
	get_slice(quater, myslice, BlockSize, __det_x, __det_y, 0);
	
	


	// test performance
	float estime = cuda_return_time();
	printf("use %.5f ms\n", estime);


	// write
	fp = fopen("tst_merge.bin", "wb");
	fwrite(myslice, sizeof(float), pat_s[0]*pat_s[1], fp);
	fclose(fp);


	// free & free cuda
	free_cuda_all();
	free(mymodel);
	free(model_2);
	free(merge_w);
	free(det);
	free(mask);
	free(myslice);

}