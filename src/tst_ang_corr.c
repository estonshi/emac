#include "base.h"
#include "cuda_funcs.h"

//int pat_s[2];
//int stoprad;
int __stoprad;
uint32 __det_x, __det_y;
float __dataset_mean_count;


int main(int argc, char** argv){
	char fn[999], mask_fn[999], vol_fn[999], det_fn[999], quat_fn[999], line[999];
	int i, c, gpu;
	int bins;
	gpu = 0;
	__stoprad = 15;
	int num_quat = 14553;
	int __qmax_len;
	int Blocksize = 16;

	int line_num;

	while( (c = getopt(argc, argv, "f:x:y:p:q:d:z:u:n:m:g:t:h")) != -1 ){
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
				if(bins<=0){
					printf("[error] partition bins <= 0\n");
					exit(1);
				}
				if(bins%2 != 0) bins++;
				break;
			case 'q':
				__qmax_len = atoi(optarg);
				break;
			case 'd':
				strcpy(vol_fn, optarg);
				break;
			case 'z':
				strcpy(det_fn, optarg);
				break;
			case 'u':
				strcpy(quat_fn, optarg);
				break;
			case 'n':
				num_quat = atoi(optarg);
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
				printf("         -m [mask file]\n");
				printf("         -x [pattern pat_s[0]]\n");
				printf("         -y [pattern pat_s[1]]\n");
				printf("         -q [3D model size]\n");
				printf("         -p [number of partition bins]\n");
				printf("         -d [volume model file]\n");
				printf("         -z [detector mapping file]\n");
				printf("         -u [quaternion file]\n");
				printf("         -n [number of quaternions]\n");
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
	float *det, *mymodel, *quater;
	float *result = (float*) malloc(bins*bins*sizeof(float));
	float *ac_diff = (float*) malloc(num_quat*sizeof(float));
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

	__dataset_mean_count = 0;
	for(i=0; i<pat_s[0]*pat_s[1];i++){
		__dataset_mean_count += signalx[i];
	}

	// mask
	fp = fopen(mask_fn, "rb");
	if(fp == NULL){
		printf("Error. Do not load mask\n");
		return -1;
	}
	fread(mask, sizeof(int), pat_s[1]*pat_s[0], fp);
	fclose(fp);

	// model_1
	mymodel = (float*) malloc(__qmax_len*__qmax_len*__qmax_len*sizeof(float));
	fp = fopen(vol_fn, "rb");
	if( fp == NULL ){
		printf("[error] Your data file is invalid.\n");
		return -1;
	}
	fread(mymodel, sizeof(float), __qmax_len*__qmax_len*__qmax_len, fp);
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


	// fake num_mask_ron, model_2 and merge_w. All set 0
	float *model_2, *merge_w;
	model_2 = (float*) calloc(__qmax_len*__qmax_len*__qmax_len, sizeof(float));
	merge_w = (float*) calloc(__qmax_len*__qmax_len*__qmax_len, sizeof(float));
	int num_mask_ron[2] = {0,0};


	// init gpu var
	gpu_var_init((int)__det_x, (int)__det_y, center, num_mask_ron, __qmax_len, (int)__stoprad, 
										det, mask, mymodel, model_2, merge_w, bins);




	// test performance
	cuda_start_event();

	// calculate pattern ac map, save in GPU buffer and also return to "result"
	do_angcorr(bins, signalx, result, pat_s[0], pat_s[1], Blocksize, false);

	float tmp[4];

	for(i=0;i<num_quat;i++){

		// quaternion
		tmp[0] = quater[i*4];
		tmp[1] = quater[i*4+1];
		tmp[2] = quater[i*4+2];
		tmp[3] = quater[i*4+3];

		// slicing from model_1, save in GPU buffer
		get_slice(tmp, NULL, Blocksize, pat_s[0], pat_s[1], 1);

		// calculate model slice ac map, save in GPU buffer
		do_angcorr(bins, NULL, NULL, pat_s[0], pat_s[1], Blocksize, true);

		// compare ac maps and output differences
		ac_diff[i] = comp_angcorr(bins, NULL, NULL, Blocksize);
		
	}

	// test performance
	float estime = cuda_return_time();
	printf("use %.5f ms to slicing %d pattern(s) and compare angular correlations\n", estime, 14553);




	//save ac_pattern
	fp = fopen("./output/tst_ac_pattern.bin", "wb");
	fwrite(result, sizeof(float), bins*bins, fp);
	fclose(fp);
	// save ac_diff
	fp = fopen("./output/tst_ac_diff.bin", "wb");
	fwrite(ac_diff, sizeof(float), num_quat, fp);
	fclose(fp);


	// free
	free(signalx);
	free(result);
	free(ac_diff);
	free(mask);
	free(det);
	free(quater);
	free(mymodel);
	free(model_2);
	free(merge_w);
	free_cuda_all();

}