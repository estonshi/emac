#include "main.h"



int main(int argc, char** argv){

	//
	if( __MPIRANK == 0 )
		time(&program_start);
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &__MPIRANK);
	MPI_Comm_size(MPI_COMM_WORLD, &__NUMPROC);

	/*
	Parse Parameters
	*/
	resume = false;
	gpu_id = 0;
	BlockSize = 16;
	//num_threads = omp_get_max_threads();
	num_threads = 1;
	while( (c = getopt(argc, argv, "c:i:g:b:t:rh")) != -1 ){
		switch (c){
			case 'c':
				strcpy(config_file, optarg);
				break;
			case 'i':
				i = atoi(optarg);
				if(i>0)
					__iter_all = i;
				else{
					printf("[ERROR] iterations should be positive integer\n");
					MPI_Finalize();
					exit(-1);
				}
				break;
			case 'g':
				//gpu_id = atoi(optarg);
				token = strtok(optarg,",");
				i = 0;
				while(i != __MPIRANK){
					token = strtok(NULL, ",");
					i++;
					if(token == NULL){
						printf("[ERROR] The number of given GPU(s) is smaller than MPI size.\n");
						MPI_Finalize();
						exit(-1);
					}
				}
				gpu_id = atoi(token);
				break;
			case 'b':
				BlockSize = atoi(optarg);
				break;
			case 't':
				num_threads = atoi(optarg);
				break;
			case 'r':
				resume = true;
				break;
			case 'h':
				printf("\nMain function to start emac program");
				printf("\nOptions:");
				printf("\n        -c [config_file] : config file path");
				printf("\n        -i [iteration]   : number of iterations");
				printf("\n        -g [gpu id]      : which GPUs to use, e.g. '0,1,2' ,default is 0");
				printf("\n        -b [Block Size]  : 2D block size used in GPU calculation, default = 16");
				printf("\n        -t [num_threads] : number of parallel threads, default = 1");
				printf("\n        -r               : if given, then resume from last iteration");
				printf("\n");
				return 0;
			default:
				printf("\nDo nothing. Exit.");
				return 0;
		}
	}


	/*
	Initiation
	*/
	printf("Submit rank %d to GPU %d\n", __MPIRANK, gpu_id);
	if( __MPIRANK == 0 )
		printf("System setup ...\n");

	// openmp
	omp_set_num_threads(num_threads);

	// host set up
	if(__iter_all <= 0){
		printf("[ERROR] use [-i] option to provide number of iterations\n");
		exit(-1);
	}
	succeed = setup(config_file, resume, __MPIRANK);
	if(!succeed){
		free_all();
		MPI_Finalize();
		exit(-1);
	}
	// bcast __model_1 if __NUMPROC > 1 and resume == false
	if(__NUMPROC > 1 && resume == false){
		MPI_Bcast(__model_1, (int)__qmax_len*__qmax_len*__qmax_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	pattern = (float*) calloc((int)__det_x*__det_y, sizeof(float*));

	PROB_MIN = 1.0 / __quat_num;
	Q_prob = 1.0 / __quat_num;
	KL_entropy = 0;
	tmp_model = NULL;

	// GPU
	setDevice(gpu_id);
	gpu_var_init((int)__det_x, (int)__det_y, __center_p, __num_mask_ron, (int)__qmax_len, (int)__ron, 
				(int)__quat_num, __quat, __det, __mask, __model_1, __model_2, __merge_w, (int)__ang_corr_grid);


	/*
	main loop
	*/
	// __iter_now start from 1
	for( ; __iter_now <= __iter_all; __iter_now++)
	{

		time(&loop_start);
		thisp = __dataset;
		if(__MPIRANK == 0){
			printf(">>> Start iteration %u / %u\n", __iter_now, __iter_all);
			printf("\t* beta = %f\n", __beta);


		/*   likelihood   */
			printf("\t* Evaluating likelihood ...\n");
		}

		for(j=0; j<__num_data; j++)
		{

			// parallel
			if(j % __NUMPROC != __MPIRANK)
				continue;

			mean_count = parse_pattern(thisp, __det_x*__det_y, __scale, pattern);
			// init pattern buffer
			memcpy_device_pattern_buf(pattern, (int)__det_x, (int)__det_y);
			total_p = 0;

			for(i=0; i<__quat_num; i++)
			{

				// slicing from model_1
				get_slice(i, NULL, BlockSize, (int)__det_x, (int)__det_y, 1);

				// calculate likelihood
				__P_jk[i + j*__quat_num] = calc_likelihood(__beta, NULL, NULL, (int)__det_x, (int)__det_y);
				total_p += __P_jk[i + j*__quat_num];

			}

			if(total_p/__quat_num < 1e-10){
				printf("[Error] likelihood is too small, give smaller beta value ! Terminated.\n");
				MPI_Finalize();
				exit(-1);
			}

			#pragma omp parallel for schedule(static,1) reduction(+:KL_entropy) private(prob_tmp)
			for(i=0; i<__quat_num; i++){
				prob_tmp =  __P_jk[i + j*__quat_num] / total_p;
				KL_entropy += prob_tmp * log(prob_tmp/Q_prob);
				__P_jk[i + j*__quat_num] = prob_tmp;
			}
			#pragma omp barrier

			thisp = thisp->next;
			if(j % (int)(__num_data/3.0) == 0)
				printf("\t  progress: %.1f % \n", (float)j/__num_data*100);

		}

		// (all) reduce __P_jk
		MPI_Allreduce(MPI_IN_PLACE, __P_jk, __num_data*__quat_num, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		// refresh thisp pointer
		thisp = NULL;

		/*  maximization  */
		if( __MPIRANK == 0 )
			printf("\t* Doing maximization ...\n");

		rescale_model_2 = 0;

		for(i=0; i<__quat_num; i++){

			// parallel
			if(i % __NUMPROC != __MPIRANK)
				continue;

			memcpy_device_slice_buf(NULL, (int)__det_x, (int)__det_y);
			total_p = 0;
			total_mean_count = 0;
			thisp = __dataset;

			for(j=0; j<__num_data; j++){

				prob_tmp = __P_jk[i + j*__quat_num];
				if(prob_tmp < PROB_MIN) continue;

				mean_count = parse_pattern(thisp, __det_x*__det_y, __scale, pattern);
				maximization_dot(pattern, prob_tmp, (int)__det_x, (int)__det_y, NULL, BlockSize);
				total_p += prob_tmp;
				total_mean_count += (mean_count * prob_tmp);

				thisp = thisp->next;

			}

			if(total_p > 0){
				rescale_model_2 +=  (total_mean_count / total_p);
				maximization_norm(1.0/total_p, (int)__det_x, (int)__det_y, BlockSize);
				merge_slice(i, NULL, BlockSize, (int)__det_x, (int)__det_y);
			}


			if( i % (int)(__quat_num/3.0) == 0)
				printf("\t  progress: %.1f % \n", (float)i/__quat_num*100);

		}

		// download __merge_w
		// all reduce (+) __merge_w and rescale_model_2
		download_volume_from_gpu(__merge_w, (int)__qmax_len, 0);
		MPI_Allreduce(MPI_IN_PLACE, __merge_w, (int)__qmax_len*__qmax_len*__qmax_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &rescale_model_2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		// upload __merge_w
		upload_models_to_gpu(NULL, NULL, __merge_w, (int)__qmax_len);


		// merge scaling
		rescale_model_2 = __dataset_mean_count / (rescale_model_2 / __quat_num);
		merge_scaling(BlockSize, BlockSize, 1);
		download_model2_from_gpu(__model_2, (int)__qmax_len);
		// all reduce (+) __model_2
		MPI_Allreduce(MPI_IN_PLACE, __model_2, (int)__qmax_len*__qmax_len*__qmax_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);


		thisp = NULL;
		time(&loop_end);
		diff_t = difftime(loop_start, loop_end);



		/* evaulate changes & rescale __model_2 */
		if( __MPIRANK == 0 )
			printf("\t* Result evaulation ...\n");

		diff_model = 0;
		model_norm = 0;

		if(__scale){

			#pragma omp parallel for schedule(static,1) reduction(+:model_norm,diff_model)
			for(i=0; i<__qmax_len*__qmax_len*__qmax_len; i++){

				model_norm += powf(__model_1[i], 2);
				__model_2[i] *= rescale_model_2;
				diff_model += powf(__model_2[i]-__model_1[i], 2);

			}
			#pragma omp barrier

		}
		else{

			#pragma omp parallel for schedule(static,1) reduction(+:model_norm,diff_model)
			for(i=0; i<__qmax_len*__qmax_len*__qmax_len; i++){

				model_norm += powf(__model_1[i], 2);
				diff_model += powf(__model_2[i]-__model_1[i], 2);

			}
			#pragma omp barrier

		}

		diff_model = sqrt(diff_model/model_norm);

		if( __MPIRANK == 0 )
			printf("\t\trelative rmsd change : %lf\n", diff_model);

		if(__scale && __MPIRANK == 0)
			printf("\t\tscaling factor       : %f\n", rescale_model_2);

		/* evaulate KL divergence of probabilities  */
		KL_entropy /= __num_data;

		if( __MPIRANK == 0 )
			printf("\t\tprobs KL-divergence  : %lf\n", KL_entropy);

		/* exchange __model_1 & __model_2   */
		upload_models_to_gpu(__model_2, NULL, NULL, (int)__qmax_len);
		reset_model((int)__qmax_len, 2);
		tmp_model = __model_1;
		__model_1 = __model_2;
		__model_2 = tmp_model;
		tmp_model = NULL;


		/* write log & update beta  */
		// __iter_now start from 1
		if(__iter_now % __beta_jump == 0){
			__beta *= __beta_mul;
		}

		if( __MPIRANK == 0 ){
			// write log
			write_log(__iter_now, (float)diff_t, (float)diff_model, __beta, __quat_num, KL_entropy);

			// save info
			sprintf(line, "%s/info/info_%.3u.txt", __output_dir, __iter_now);
			fp = fopen(line, "w");
			if(__scale)
				fprintf(fp, "%f, %lf\n", rescale_model_2, KL_entropy);
			else
				fprintf(fp, "1.0, %lf\n", KL_entropy);
			fclose(fp);

			// save model
			sprintf(line, "%s/model/model_%.3u.bin", __output_dir, __iter_now);
			fp = fopen(line, "w");
			fwrite(__model_1, sizeof(float), __qmax_len*__qmax_len*__qmax_len, fp);  // __model_1 & __model_2 are exchanged
			fclose(fp);

			// save probabilities
			sprintf(line, "%s/probs/probs_%.3u.bin", __output_dir, __iter_now);
			fp = fopen(line, "w");
			fwrite(__P_jk, sizeof(float), __quat_num*__num_data, fp);
			fclose(fp);
			if(__iter_now % 5 == 0 || __iter_now == __iter_all || __iter_now == 1){
				rdata *rsort = (rdata*)malloc(__quat_num * sizeof(rdata));
				rdata buffer;
				sprintf(line, "%s/probs/probs_%.3u_top10.txt", __output_dir, __iter_now);
				fp = fopen(line, "w");
				for(j=0; j<__num_data; j++){
					for(i=0; i<__quat_num; i++){
						rsort[i].data = __P_jk[i + j*__quat_num];
						rsort[i].index = i;
					}
					qsort(rsort, __quat_num, sizeof(rsort[0]), cmp);
					for(i=0; i<10; i++){
						buffer = rsort[i];
						fprintf(fp, "%f,%d\n", buffer.data, buffer.index);
					}
				}
				fclose(fp);
				free(rsort);
			}
		}

		// refresh criterion
		KL_entropy = 0;
		
	}


	/*
	free
	*/
	free(pattern);
	free_cuda_all();
	free_all();

	if( __MPIRANK == 0 ){
		time(&program_end);
		diff_t = difftime(program_end, program_start);
		printf("Total used time : %lf s\n", diff_t);
	}

	MPI_Finalize();
}