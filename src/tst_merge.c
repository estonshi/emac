#include "base.h"
#include "cuda_funcs.h"
 

uint32 __qmax_len;
uint32 __det_x, __det_y;
 


int main(int argc, char** argv){
	char fn[999], mask_fn[999], det_fn[999], quat_fn[999], line[999];;
	int c, gpu;
	int BlockSize = 16;
	int num_quat;
	gpu = 0;

	float *det, *mymodel, *model_2, *merge_w, *myslice, *quater;
	
	int *mask;

	int pat_s[2];
	float center[2];

	int i, line_num;

	FILE* fp;

	while( (c = getopt(argc, argv, "f:d:x:y:l:m:q:g:z:h")) != -1 ){
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
			case 'q':
				strcpy(quat_fn, optarg);
				break;
			case 'g':
				gpu = atoi(optarg);
				break;
			case 'z':
				num_quat = atoi(optarg);
				break;
			case 'h':
				printf("options:\n");
				printf("         -f [signal file]\n");
				printf("         -d [detector file]\n");
				printf("         -x [pattern pat_s[0]]\n");
				printf("         -y [pattern pat_s[1]]\n");
				printf("         -l [length of volume]\n");
				printf("         -m [mask file]\n");
				printf("         -q [quaternion file]\n");
				printf("         -g [GPU number]\n");
				printf("         -z [number of quaternions]\n");
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

	// read quaternion file
	quater = (float*) malloc(num_quat*sizeof(float)*4);
	fp = fopen(quat_fn, "r");
	if(fp == NULL){
		printf("[ERROR] Your quaternion file is invalid.\n");
		return false;
	}	
	line_num = 0;

	while (fgets(line, 999, fp) != NULL) {
		if(line_num < num_quat){
			quater[line_num*4] = atof(strtok(line, " \n"));
			quater[line_num*4+1] = atof(strtok(NULL, " \n"));
			quater[line_num*4+2] = atof(strtok(NULL, " \n"));
			quater[line_num*4+3] = atof(strtok(NULL, " \n"));
			line_num ++;
		}
	}
	fclose(fp);


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
	line_num = 0;
	
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


	// fake num_mask_ron, set to 0
	int num_mask_ron[2] = {0,0};

	// init gpu var
	gpu_var_init((int)__det_x, (int)__det_y, center, num_mask_ron, (int)__qmax_len, 20, 
									det, mask, mymodel, model_2, merge_w, 256);


	// test performance
	cuda_start_event();

	float tmp[4];
	myslice = (float*) malloc(pat_s[0]*pat_s[1]*sizeof(float));

	for(i=0; i<num_quat; i++)
	{

		tmp[0] = quater[i*4];
		tmp[1] = quater[i*4+1];
		tmp[2] = quater[i*4+2];
		tmp[3] = quater[i*4+3];

		// slicing from model_1
		get_slice(tmp, myslice, BlockSize, pat_s[0], pat_s[1], 1);

		// do somthing
		// ...

		// merge to model_2	
		merge_slice(tmp, myslice, BlockSize, pat_s[0], pat_s[1]);

	}

	// scaling
	merge_scaling(BlockSize, BlockSize);

	// test performance
	float estime = cuda_return_time();
	printf("use %.5f ms for slicing & merging %d pattern(s)\n", estime, num_quat);


	// download model_2 from gpu
	download_model2_from_gpu(model_2, (int)__qmax_len);

	// write
	fp = fopen("./output/tst_slicing.bin", "wb");
	fwrite(myslice, sizeof(float), pat_s[0]*pat_s[1], fp);
	fclose(fp);

	fp = fopen("./output/tst_merging.bin", "wb");
	fwrite(model_2, sizeof(float), __qmax_len*__qmax_len*__qmax_len, fp);
	fclose(fp);


	// free & free cuda
	free_cuda_all();
	free(mymodel);
	free(model_2);
	free(merge_w);
	free(det);
	free(mask);
	free(myslice);
	free(quater);

}