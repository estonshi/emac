#include "params.h"


bool load_det_info(char* det_file, int* mask, float* det){
	// read det file
	FILE* fp;
	fp = fopen(det_file, "r");
	if(fp == NULL){
		printf("[ERROR] Your detector file is invalid.\n");
		return false;
	}
	char line[999], *token;
	int line_num = 0;
	uint32 q_max_len;
	__num_mask_ron[0] = 0;
	__num_mask_ron[1] = 0;

	while (fgets(line, 999, fp) != NULL) {
		if(line_num < __det_x * __det_y){
			det[line_num*3] = atof(strtok(line, " \n"));
			det[line_num*3+1] = atof(strtok(NULL, " \n"));
			det[line_num*3+2] = atof(strtok(NULL, " \n"));
			mask[line_num] = atoi(strtok(NULL, " \n"));
			if(mask[line_num] == 1) __num_mask_ron[1]++;
			else if(mask[line_num] == 2) __num_mask_ron[0]++;
			line_num ++;
		}
		else
			q_max_len = (uint32)atoi(strtok(line, "\n"));
	}

	fclose(fp);
	// property from params.h
	__qmax_len = 2 * q_max_len + 1;
	__center_q[0] = q_max_len;
	__center_q[1] = q_max_len;
	__center_q[2] = q_max_len;

	return true;
}


// iter_num start from 1 !!!!
void write_log(uint32 iter_num, float used_time, float rms_change, float beta, uint32 num_rot, float KL_entropy){
	FILE* fp;
	fp = fopen(__log_file, "a");
	fprintf(fp, "%-10u %-12.2f %-12.3e %-12.3e %-12u %-10.5f\n", iter_num, used_time, rms_change, KL_entropy, num_rot, beta);
	fclose(fp);
}


float read_log(char* read_item, int iter_num){
	int i;
	if(iter_num == 0){
		printf("[ERROR] Iter number start from 1 or give -1 to ignore.\n");
		return -1;
	}

	FILE* fp;
	fp = fopen(__log_file, "r");
	if( fp == NULL ){
		printf("[ERROR] The log file is invalid.\n");
		return -1;
	}

	float info[6];
	float returned;

	int index = 0;
	if( strcmp(read_item, "iter_num") == 0 ) index = 0;
	else if( strcmp(read_item, "used_time") == 0 ) index = 1;
	else if( strcmp(read_item, "rms_change") == 0 ) index = 2;
	else if( strcmp(read_item, "KL_divg") == 0 ) index = 3;
	else if( strcmp(read_item, "beta") == 0 ) index = 5;
	else{
		printf("[ERROR] Log file doesn't have information about '%s'. Return -1.\n", read_item);
		return -1;
	}

	char line[999], *token;
	while( fgets(line, 999, fp) != NULL){
		if( strstr(line, "iter_num") != NULL ) break;
	}

	int iter_total = 0;
	while( fgets(line, 999, fp) != NULL ){
		token = strtok(line, " \n");
		iter_total = atoi(token);
		if(iter_num != iter_total) continue;
		for(i=0; i<index; i++){
			token = strtok(NULL, " \n");
		}
		returned = atof(token);
		break;
	}

	fclose(fp);
	if(iter_total == 0){
		printf("[ERROR] Log file doesn't have any information yet. Return -1.\n");
		return -1;
	}
	if(iter_num < 0) returned = (float)iter_total;
	
	return returned;
}


bool load_model(char* model_file, uint32 size, float* model){
	// read binary model file
	FILE* fp;
	fp = fopen(model_file, "rb");
	if(fp == NULL){
		printf("[ERROR] Your model file is invalid.\n");
		return false;
	}

	fread(model, sizeof(float), size, fp);

	fclose(fp);
	return true;
}


bool setup(char* config_file, bool resume, int mpi_rank){

	int i;
	// read config file
	FILE* fp;
	fp = fopen(config_file, "r");
	if(fp == NULL){
		printf("[ERROR] Your config file is invalid. Exit.\n");
		return false;
	}
	char line[999], *token;

	int temp_value_i;
	char init_model[999];
	int quat_lev;
	uint32 q_max_len;

	__data_path[0] = '\0';
	__det_path[0] = '\0';
	__quat_path[0] = '\0';
	__output_dir[0] = '\0';
	init_model[0] = '\0';

	while (fgets(line, 999, fp) != NULL) {
		token = strtok(line, " =") ;
		if (token[0] == '#' || token[0] == '\n') {
			continue ;
		}
		else if (token[0] == '[') {
			token = strtok(token, "[]") ;
			if (strcmp(token, "input") == 0 ||
			    strcmp(token, "adjust") == 0 ||
			    strcmp(token, "recon") == 0 ||
			    strcmp(token, "output") == 0)
				temp_value_i = 1 ;
			else
				temp_value_i = 0 ;
			continue ;
		}
		if (!temp_value_i)
			continue ;

		// [input]
		if (strcmp(token, "emac_data") == 0)
			strcpy(__data_path, strtok(NULL, " = \n"));
		else if (strcmp(token, "quat") == 0)
			strcpy(__quat_path, strtok(NULL, " = \n"));
		else if (strcmp(token, "quat_lev") == 0){
			quat_lev = atoi(strtok(NULL, " = \n"));
			if(quat_lev <=0 ){
				printf("[ERROR] Your given 'quat_lev' should be an positive integer. Exit.\n");
				return false;
			}
			// func from gen_quat.h
			__quat_num = (uint32)cal_quat_num(quat_lev);
		}
		else if (strcmp(token, "size") == 0) {
			token = strtok(NULL, " = \n");
			__det_x = (uint32)atoi(strtok(token, " ,\n"));
			__det_y = (uint32)atoi(strtok(NULL, " ,\n"));
		}
		else if (strcmp(token, "center") == 0){
			token = strtok(NULL, " = ");
			__center_p[0] = atof(strtok(token, " ,\n"));
			__center_p[1] = atof(strtok(NULL, " ,\n"));
		}
		// [adjust]
		else if (strcmp(token, "scaling") == 0){
			token = strtok(NULL, " = \n");
			if( strcmp(token, "True") || strcmp(token, "true") )
				__scale = true;
			else
				__scale = false;
		}
		else if (strcmp(token, "ron") == 0){
			__ron = (uint32)atoi(strtok(NULL, " = \n"));
		}
		// [recon]
		else if (strcmp(token, "init_model") == 0){
			token = strtok(NULL, " = \n");
			if( strcmp(token, "None") != 0 && strcmp(token, "none") != 0 && strcmp(token, "NONE") != 0)
				strcpy(init_model, token);
		}
		else if (strcmp(token, "beta") == 0){
			token = strtok(NULL, " = \n");
			__beta = atof(strtok(token, " ,\n"));
			__beta_jump = atoi(strtok(NULL, " ,\n"));
			__beta_mul = atof(strtok(NULL, " ,\n"));
		}
		else if (strcmp(token, "ang_corr_grid") == 0){
			token = strtok(token, " = \n");
			if( strcmp(token, "None") == 0 || strcmp(token, "none") == 0 || strcmp(token, "NONE") == 0 )
				__ang_corr_grid = 0;
			else
				__ang_corr_grid = (uint32)atoi(token);
		}
		// [output]
		else if (strcmp(token, "det_q") == 0)
			strcpy(__det_path, strtok(NULL, " = \n"));
		else if (strcmp(token, "temp") == 0)
			strcpy(__output_dir, strtok(NULL, " = \n"));
	}

	fclose(fp);

	if( mpi_rank == 0 ){
		if( !resume )
			printf("[Initial]\n");
		else
			printf("[Resuming]\n");
		printf("Number of rotations is        : %u\n", __quat_num);
	}

	// load data, func from emac_data.h
	uint32 size_x, size_y;
	if( !load_emac_prop(__data_path, &__num_data, &size_x, &size_y) )
		return false;
	if (size_x != __det_x || size_y != __det_y){
		__det_x = size_x;
		__det_y = size_y;
		printf("[WARNING] Your given pattern size is invalid, change to (%u,%u)\n", size_x, size_y);
	}

	__dataset = (emac_pat*) malloc(sizeof(emac_pat));
	__dataset_mean_count = load_emac_dataset(__data_path, __dataset);
	if( __dataset_mean_count == -1 ) return false;

	if( mpi_rank == 0 ){
		printf("Number of patterns is         : %u\n", __num_data);
		printf("Dataset mean photon count is  : %f\n", __dataset_mean_count);
		printf("Need scaling                  : %s\n", __scale ? "true" : "false");
	}

	// generate quaternion, func from gen_quat.h
	__quat = (float*) malloc(__quat_num * 4 * sizeof(float));
	gen_quaternions(quat_lev, 0, __quat);   // mode must be 0, as mode 1 uses random number and cannot be paralleled

	// load detector pixels' qinfo, func from params.h
	__mask = (int*) malloc(__det_x * __det_y * sizeof(int));
	__det = (float*) malloc(__det_x * __det_y * 3 * sizeof(float));
	if( !load_det_info(__det_path, __mask, __det) ) 
		return false;

	if( mpi_rank == 0 ){
		printf("Merged volume size is         : %u x %u x %u\n", __qmax_len, __qmax_len, __qmax_len);
	}

	// initiate model_1, model_2 and merge_w
	__model_1 = (float*) malloc(__qmax_len*__qmax_len*__qmax_len*sizeof(float));
	__model_2 = (float*) calloc(__qmax_len*__qmax_len*__qmax_len, sizeof(float));
	__merge_w = (float*) malloc(__qmax_len*__qmax_len*__qmax_len*sizeof(float));

	if( !resume ){

		if(init_model[0] != '\0'){
			if( mpi_rank == 0 )
				printf("Using initial model           : %s\n", init_model);
			if( !load_model(init_model, __qmax_len*__qmax_len*__qmax_len, __model_1) )
				return false;
		}
		else{
			if( mpi_rank == 0 )
				printf("Using random initial model\n");
			srand(time(NULL));
			for(i=0; i<__qmax_len*__qmax_len*__qmax_len; i++){
				__model_1[i] = (float) rand() / RAND_MAX * 2 * __dataset_mean_count;
				__merge_w[i] = 1.0f;
			}
		}

	}

	// initiate __P_jk, uses calloc for reduction(+) after parallization
	__P_jk = (float*) calloc(__num_data*__quat_num, sizeof(float));


	if( !resume ){

		// initiate iter 
		__iter_now = 1;

		if( mpi_rank == 0 ){

			// mkdir
			sprintf(line, "%s/info", __output_dir);
			mkdir(line, 0750);
			sprintf(line, "%s/probs", __output_dir);
			mkdir(line, 0750);
			sprintf(line, "%s/model", __output_dir);
			mkdir(line, 0750);

			// write initial information
			// write scaling
			sprintf(line, "%s/info/scaling_init.txt", __output_dir);
			if(__scale){
				fp = fopen(line, "w");
				emac_pat* thisp = __dataset;
				for(i=0; i<__num_data; i++){
					fprintf(fp, "%.4f\n", thisp->scale_factor);
					thisp = thisp->next;
				}
				fclose(fp);
			}
			// write model
			sprintf(line, "%s/model/model_000.bin", __output_dir);
			fp = fopen(line, "wb");
			fwrite(__model_1, sizeof(float), __qmax_len*__qmax_len*__qmax_len, fp);
			fclose(fp);
			// write log
			sprintf(__log_file, "%s/log.txt", __output_dir);
			fp = fopen(__log_file, "w");
			fprintf(fp, "[Initiate]\n");
			fprintf(fp, "Number of rotations   :   %u\n", __quat_num);
			fprintf(fp, "Number of patterns    :   %u\n", __num_data);
			fprintf(fp, "Mean photon count     :   %f\n", __dataset_mean_count);
			if(__scale)
				fprintf(fp, "Need Scaling          :   yes\n");
			else
				fprintf(fp, "Need Scaling          :   no\n");
			fprintf(fp, "Merged volume size    :   %u x %u x %u\n", __qmax_len, __qmax_len, __qmax_len);
			if(init_model[0] != '\0')
				fprintf(fp, "Initial model         :   %s\n", init_model);
			else
				fprintf(fp, "Initial model         :   random\n");
			fprintf(fp, "\n[Iterations]\n");
			fprintf(fp, "%-10s %-12s %-12s %-12s %-12s %-10s\n", "iter_num", "used_time", "rms_change", "KL_divg", "num_rot", "beta");
			fclose(fp);

			printf("Log file is                   : %s\n", __log_file);

		}
	}

	else{

		// read info from last iteration
		if( mpi_rank == 0 )
			sprintf(__log_file, "%s/log.txt", __output_dir);

		// iter number
		float temp;
		temp = read_log("iter_num",-1);
		if(temp <= 0) return false;
		else __iter_now = (uint32)(temp+0.1) + 1;
		if( mpi_rank == 0 )
			printf("Resume from iteration         : %u\n", __iter_now);

		// model
		if( mpi_rank == 0 )
			sprintf(init_model, "%s/model/model_%.3u.bin", __output_dir, __iter_now - 1);
		if( !load_model(init_model, __qmax_len*__qmax_len*__qmax_len, __model_1) )
			return false;

		for(i=0; i<__qmax_len*__qmax_len*__qmax_len; i++){
			__merge_w[i] = 1.0f;
		}

		// beta
		temp = read_log("beta", __iter_now - 1);
		if(temp <= 0) return false;
		else{
			if( __iter_now != 1 && (__iter_now-1) % __beta_jump == 0)
				__beta = temp * __beta_mul;
			else
				__beta = temp;
		}

		// write info to log
		fp = fopen(__log_file, "a");
		fprintf(fp, "(Resuming)\n");
		fclose(fp);

	}

	// Done.
	printf("\n");
	return true;

}


void free_all(){
	// free model
	free(__model_1);
	free(__model_2);
	free(__merge_w);
	// free detector
	free(__det);
	// free quaternions
	free(__quat);
	// free mask
	free(__mask);
	// free __P_jk
	free(__P_jk);
	// free dataset
	emac_pat *thisp = __dataset;
	emac_pat *nextp;
	while(thisp != NULL){
		nextp = thisp->next;
		free(thisp);
		thisp = nextp;
	}
}


/*
int main(int argc, char** argv){
	char config_file[999];

	int i, c;
	bool resume = false;
	while( (c = getopt(argc, argv, "c:rh")) != -1 ){
		switch (c){
			case 'c':
				strcpy(config_file, optarg);
				break;
			case 'r':
				resume = true;
				break;
			case 'h':
				printf("\nThis function is used to test setup()");
				printf("\nOptions:");
				printf("\n        -c [config_file] : config file path");
				printf("\n        -r  : (resume from last iteration)");
				printf("\n");
				return 0;
			default:
				printf("\nDo nothing. Exit.");
				return 0;
		}
	}

	bool succeed = setup(config_file, resume);
	if(!succeed){
		free_all();
		return -1;
	}

	for(i=1; i<=3; i++){
		write_log(__iter_now, 300.0f, 3.3f, __beta, __quat_num, 1.45);
		__iter_now ++;
		if( __iter_now != 1 && (__iter_now-1) % __beta_jump == 0)
			__beta *= __beta_mul;
	}

	float info = read_log("rms_change", 2);
	printf("test : rms_change of iteration 2 is %f\n", info);

	free_all();
}

*/