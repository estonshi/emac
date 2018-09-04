#include "base.h"
#include "emac_data.h"
#include "cuda_funcs.h"
 

uint32 __qmax_len;
uint32 __det_x, __det_y;
uint32 __num_data;
int __num_mask_ron[2];
float __beta;


int main(int argc, char** argv){
	char fn[999], mask_fn[999], det_fn[999], quat_fn[999], line[999], pat_fn[999];
	int c, gpu;
	int BlockSize = 16;
	int num_quat;
	gpu = 0;

	float *det, *mymodel, *model_2, *merge_w, *myslice, *quater;
	float *pattern, *P_jk;
	
	int *mask;

	int pat_s[2];
	float center[2];

	int i, j, line_num;

	FILE* fp;

	while( (c = getopt(argc, argv, "f:d:b:l:m:q:g:z:p:h")) != -1 ){
		switch(c){
			case 'f':
				strcpy(fn, optarg);
				break;
			case 'd':
				strcpy(det_fn, optarg);
				break;
			case 'b':
				__beta = atof(optarg);
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
			case 'p':
				strcpy(pat_fn, optarg);
				break;
			case 'h':
				printf("options:\n");
				printf("         -f [signal file]\n");
				printf("         -d [detector file]\n");
				printf("         -b [beta value]\n");
				printf("         -l [length of volume]\n");
				printf("         -m [mask file]\n");
				printf("         -q [quaternion file]\n");
				printf("         -g [GPU number]\n");
				printf("         -z [number of quaternions]\n");
				printf("         -p [emac dataset file]\n");
				printf("         -h help\n");
				return 0;
			default:
				printf("Do nothing.\n");
				break;
		}
	}

	setDevice(gpu);

	// dataset
	emac_pat* dataset = (emac_pat*) malloc(sizeof(emac_pat));

	load_emac_prop(pat_fn, &__num_data, &__det_x, &__det_y);
	float mean_count = load_emac_dataset(pat_fn, dataset);

	pattern = (float*) calloc((int)__det_x*__det_y, sizeof(float*));
	mean_count = parse_pattern(dataset, __det_x*__det_y, true, pattern);


	// other init

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
	__num_mask_ron[0] = 0;
	__num_mask_ron[1] = 0;
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
			if(mask[line_num] == 1) __num_mask_ron[1]++;
			else if(mask[line_num] == 2) __num_mask_ron[0]++;
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
	gpu_var_init((int)__det_x, (int)__det_y, center, __num_mask_ron, (int)__qmax_len, 20, 
									det, mask, mymodel, model_2, merge_w, 256);
	// init pattern buffer
	memcpy_device_pattern_buf(pattern, __det_x, __det_y);


	// test performance
	cuda_start_event();

	float tmp[4], tmp_mean_count = 0, total_p = 0;
	myslice = (float*) malloc(pat_s[0]*pat_s[1]*sizeof(float));
	P_jk = (float*) malloc(num_quat*sizeof(float));

	for(i=0; i<num_quat; i++)
	{

		tmp[0] = quater[i*4];
		tmp[1] = quater[i*4+1];
		tmp[2] = quater[i*4+2];
		tmp[3] = quater[i*4+3];

		// slicing from model_1
		get_slice(tmp, NULL, BlockSize, pat_s[0], pat_s[1], 1);

		
		/*
		// calc mean count of slice
		for(j=0; j<__det_x*__det_y; j++){
			tmp_mean_count += myslice[j];
		}
		for(j=0; j<__det_x*__det_y; j++){
			myslice[j] *= (mean_count/tmp_mean_count);
		}
		*/// No need to do scaling here
		// scaling is done to __model_2 at the end of each iteration


		// calculate likelihood
		calc_likelihood(__beta, NULL, NULL, pat_s[0], pat_s[1], &P_jk[i]);
		total_p += P_jk[i];
	}

	for(i=0; i<num_quat; i++){
		P_jk[i] /= total_p;
	}

	// test performance
	float estime = cuda_return_time();
	printf("use %.5f ms for slicing %d pattern(s) & calculate likelihood\n", estime, num_quat);

	// write
	fp = fopen("./output/tst_likelihood.bin", "wb");
	fwrite(P_jk, sizeof(float), num_quat, fp);
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
	free(pattern);
	free(P_jk);

	emac_pat *thisp = dataset;
	emac_pat *nextp;
	while(thisp != NULL){
		nextp = thisp->next;
		free(thisp);
		thisp = nextp;
	}

}