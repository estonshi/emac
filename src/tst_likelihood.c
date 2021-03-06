#include "base.h"
#include "emac_data.h"
#include "cuda_funcs.h"
extern void gpu_dataset_init(emac_pat *dataset, int num_pat);

uint32 __qmax_len;
uint32 __det_x, __det_y;
uint32 __num_data;
int __num_mask_ron[2];
float __beta;
float __PROB_MIN;


int main(int argc, char** argv){
	char fn[999], mask_fn[999], det_fn[999], quat_fn[999], line[999], pat_fn[999];
	int c, gpu;
	int BlockSize = 8;
	int num_quat;
	gpu = 0;

	float *det, *correction, *mymodel, *model_2, *merge_w, *myslice, *quater, *scaling;
	float *pattern;
	double *P_jk;
	
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
				printf("\nThis function is used to test likelihood calculation & maximization\n");
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

	__PROB_MIN = 1.0 / num_quat;

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
	correction = (float*) malloc(pat_s[0]*pat_s[1]*sizeof(float));
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
			correction[line_num] = atof(strtok(NULL, " \n"));
			if(mask[line_num] == 1) __num_mask_ron[1]++;
			else if(mask[line_num] == 2) __num_mask_ron[0]++;
			line_num ++;
		}
		else
			__qmax_len = (uint32)atoi(strtok(line, " \n"));
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

	// scaling
	scaling = (float*) malloc(__num_data*sizeof(float));
	emac_pat* thisp = dataset;
	for(i=0; i<__num_data; i++){
		scaling[i] = thisp->scale_factor;
		thisp = thisp->next;
	}
	
	// init gpu var
	gpu_var_init(pat_s[0], pat_s[1], center, __num_mask_ron, (int)__qmax_len, 20, 
				num_quat, quater, det, correction, mask, mymodel, model_2, merge_w, 256, 1);
	gpu_dataset_init(dataset, __num_data);


	double prob_tmp = 0, total_p = 0;
	float tmp = 0;
	P_jk = (double*) malloc(10*num_quat*sizeof(double));

	// calculate sum(slice)
	myslice = (float*) malloc(1234*sizeof(float));
	for(i=0; i<1234; i++) myslice[i] = i;
	tmp = slice_sum(myslice, 1, 1234, NULL);
	printf("Sum of 0~1234 is %f\n", tmp);
	prob_tmp = calc_likelihood(1, myslice, myslice, 3, 5, 1);
	printf("Likelihood between 0..14 and 0..14 is %lf\n", prob_tmp);
	free(myslice);


	/*    likelihood    */

	// test performance
	cuda_start_event();

	thisp = dataset;

	for(j=0; j<10; j++)
	{
		//mean_count = parse_pattern(thisp, pat_s[0]*pat_s[1], true, pattern);
		// init pattern buffer
		//memcpy_device_pattern_buf(pattern, pat_s[0], pat_s[1]);
		c = thisp->one_pix + thisp->mul_pix;
		parse_pattern_ongpu(j, c, pat_s[0], pat_s[1], NULL, true);
		
		total_p = 0;

		for(i=0; i<num_quat; i++)
		{

			// slicing from model_1
			get_slice(i, NULL, BlockSize, pat_s[0], pat_s[1], 1);

			// calculate likelihood
			P_jk[i + j*num_quat] = calc_likelihood(__beta, NULL, NULL, pat_s[0], pat_s[1], scaling[j]);
			total_p += P_jk[i + j*num_quat];

		}

		for(i=0; i<num_quat; i++){
			P_jk[i + j*num_quat] /= total_p;
		}

		thisp = thisp->next;
	}

	// test performance
	float estime = cuda_return_time();
	printf("use %.5f ms for evaluating probs of %d pattern(s) in %d orientations.\n", estime, 10, num_quat);

	// write
	fp = fopen("./output/tst_likelihood.bin", "wb");
	fwrite(P_jk, sizeof(double), 10*num_quat, fp);
	fclose(fp);


	/*    maximization    */

	// test performance
	cuda_start_event();

	for(i=0; i<num_quat; i++){

		thisp = dataset;
		memcpy_device_slice_buf(NULL, pat_s[0], pat_s[1]);
		total_p = 0;

		for(j=0; j<10; j++){

			prob_tmp = P_jk[i + j*num_quat];
			if(prob_tmp < __PROB_MIN) continue;

			c = thisp->one_pix + thisp->mul_pix;
			parse_pattern_ongpu(j, c, pat_s[0], pat_s[1], NULL, false);
			maximization_dot(NULL, (float)prob_tmp, pat_s[0], pat_s[1], NULL, BlockSize);

			total_p += prob_tmp;
			thisp = thisp->next;

		}

		if(total_p > 0){
			maximization_norm((float)(1.0/total_p), pat_s[0], pat_s[1], BlockSize);
			merge_slice(i, NULL, BlockSize, pat_s[0], pat_s[1]);
		}

	}

	merge_scaling(BlockSize, BlockSize, 1);

	download_model2_from_gpu(model_2, (int)__qmax_len);

	// write
	fp = fopen("./output/tst_new_model.bin", "wb");
	fwrite(model_2, sizeof(float), __qmax_len*__qmax_len*__qmax_len, fp);
	fclose(fp);


	// test performance
	estime = cuda_return_time();
	printf("use %.5f ms for maximization & merge %d new slice(s).\n", estime, 10, num_quat);




	// free & free cuda
	free_device_all();
	free(mymodel);
	free(model_2);
	free(merge_w);
	free(det);
	free(correction);
	free(mask);
	//free(myslice);
	free(quater);
	free(pattern);
	free(P_jk);

	thisp = dataset;
	emac_pat *nextp;
	while(thisp != NULL){
		nextp = thisp->next;
		free(thisp);
		thisp = nextp;
	}

}
