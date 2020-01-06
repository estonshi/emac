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
	MPI_Barrier(MPI_COMM_WORLD);
	if( __MPIRANK == 0 )
		printf("System setup ...\n\n");

	// openmp
	omp_set_num_threads(num_threads);

	// host set up
	if(__iter_all <= 0){
		printf("[ERROR] use [-i] option to provide number of iterations\n");
		exit(-1);
	}
	succeed = setup(config_file, resume, __MPIRANK);
	if(!succeed){
		free_host_global();
		MPI_Finalize();
		exit(-1);
	}
	// bcast __model_1 if __NUMPROC > 1 and resume == false
	if(__NUMPROC > 1 && resume == false){
		MPI_Bcast(__model_1, (int)__qmax_len*__qmax_len*__qmax_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	if(__scale){
		slice_intensity_r = (float*) calloc((int)__quat_num, sizeof(float));
		scaling_update_denominator = (float*) calloc((int)__num_data, sizeof(float));
	}

	Q_prob = 1.0 / __quat_num;
	KL_entropy = 0;
	MPI_Barrier(MPI_COMM_WORLD);

	// GPU init
	printf("Copying files to GPU %d ...\n", gpu_id);
	setDevice(gpu_id);
	gpu_var_init((int)__det_x, (int)__det_y, __center_p, __num_mask_ron, (int)__qmax_len, (int)__ron, (int)__quat_num, 
				__quat, __det, __correction, __mask, __model_1, __model_2, __merge_w, (int)__ang_corr_grid, num_threads);
	gpu_dataset_init(__dataset, (int)__num_data);
	free_dataset();

	MPI_Barrier(MPI_COMM_WORLD);

	/*
	main loop
	*/
	// __iter_now start from 1
	for( ; __iter_now <= __iter_all; __iter_now++)
	{

		time(&loop_start);
		// define PROB_MIN
		if( __iter_now <= __iter_all/2 )
			PROB_MIN = (2.0f*(float)(__iter_now-1)/(float)__iter_all)*(0.1f/__quat_num);
		else
			PROB_MIN = 0.1f/__quat_num;

		if( __MPIRANK == 0 ){
			printf(">>> Start iteration %u / %u\n", __iter_now, __iter_all);
			printf("\t* beta = %f\n", __beta);


		/*   likelihood   */
			printf("\t* Evaluating likelihood ...\n");
		}

		for(i=0; i<__quat_num; i++)
		{

			// parallel
			if(i % __NUMPROC != __MPIRANK)
				continue;

			// no need to loop data on host (thisp = __dataset;)
			// total_p = 0;

			// slicing from model_1
			get_slice(i, NULL, BlockSize, (int)__det_x, (int)__det_y, 1);

			// calculate sum_t{W_rt} for likelihood calculation and scaling update
			diff_model = slice_sum(NULL, (int)__det_x, (int)__det_y, NULL);
			if(__scale) slice_intensity_r[i] = diff_model;
			diff_model = diff_model * __beta / (float)(__det_x * __det_y - __num_mask_ron[0] - __num_mask_ron[1]);

			// data loop
			for(j=0; j<__num_data; j++)
			{

				threads_id = omp_get_thread_num();
				scale_tmp = __scaling_factor[j];
				c = __pix_hasphoton[j*3];

				/*
				parse_pattern_ongpu(j, c, (int)__det_x, (int)__det_y, NULL, __scale);
				__P_jk[i + j*__quat_num] = calc_likelihood(__beta, NULL, NULL, (int)__det_x, (int)__det_y, scale_tmp);
				*/

				prob_tmp = calc_likelihood_part(j, c, (int)__det_x, (int)__det_y, __beta, NULL, scale_tmp);
				__P_jk[i + j*__quat_num] = exp(prob_tmp - diff_model * scale_tmp);
				
				// total_p += __P_jk[i + j*__quat_num];
				//thisp = thisp->next;

			}
			/*
			if(total_p/__num_data < 1e-10){
				printf("[Error] Rank %d : likelihood is too small, give smaller beta value ! Terminated.\n", __MPIRANK);
				MPI_Finalize();
				exit(-1);
			}*/

			if(i % (int)(__quat_num/5.0) == 0)
				printf("\t  progress: %.1f % \n", (float)i/__quat_num*100);

		}

		// (all) reduce __P_jk
		MPI_Allreduce(MPI_IN_PLACE, __P_jk, __num_data*__quat_num, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		// reduce slice_intensity_r if __scale == True
		if(__scale)
			MPI_Allreduce(MPI_IN_PLACE, slice_intensity_r, __quat_num, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		// calc __P_jk and KL
#pragma omp parallel for schedule(static,1) reduction(+:KL_entropy) private(i,total_p,prob_tmp)
		for(j=0; j<__num_data; j++){
			total_p = 0;
			for(i=0; i<__quat_num; i++){
				total_p += __P_jk[i + j*__quat_num];
			}
			for(i=0; i<__quat_num; i++){
				prob_tmp =  __P_jk[i + j*__quat_num] / total_p;
				KL_entropy += prob_tmp * log(prob_tmp/Q_prob);
				__P_jk[i + j*__quat_num] = prob_tmp;
			}
		}
#pragma omp barrier
		
		// save likelihood
		sprintf(line, "tst_parse-l.bin");
		fp = fopen(line, "w");
		fwrite(__P_jk, sizeof(double), (int)__num_data*(int)__quat_num, fp);
		fclose(fp);
		//MPI_Finalize();exit(-1);
				

		// refresh thisp pointer
		// thisp = __dataset;

		/*  maximization  */
		if( __MPIRANK == 0 )
			printf("\t* Doing maximization ...\n");

		// refresh scaling_update_denominator
		if(__scale){
			for(j=0; j<__num_data; j++){
				scaling_update_denominator[j] = 0;
			}
		}

		for(i=0; i<__quat_num; i++){

			// parallel
			if(i % __NUMPROC != __MPIRANK)
				continue;

			// reset myslice=0 on gpu
			memcpy_device_slice_buf(NULL, (int)__det_x, (int)__det_y);
			total_p = 0;

			for(j=0; j<__num_data; j++){

				prob_tmp = __P_jk[i + j*__quat_num];
				// calculate scaling updated rule
				if(__scale)
					scaling_update_denominator[j] += (float)prob_tmp * slice_intensity_r[i];

				if((float)prob_tmp < PROB_MIN) continue;

				// parser pattern on GPU, scaling is done here
				// c = __pix_hasphoton[j*3];
				// parse_pattern_ongpu(j, c, (int)__det_x, (int)__det_y, NULL, __scale);
				// do maximization
				// maximization_dot(NULL, (float)prob_tmp, (int)__det_x, (int)__det_y, NULL, BlockSize);

				// do maximization partly
				maximization_dot_part(j, (float)prob_tmp, __pix_hasphoton[j*3+1], __pix_hasphoton[j*3+2]);

				if(__scale) total_p += prob_tmp * __scaling_factor[j];
				else total_p += prob_tmp;

			}

			if(total_p > 0){
				maximization_norm((float)(1.0/total_p), (int)__det_x, (int)__det_y, BlockSize);
				merge_slice(i, NULL, BlockSize, (int)__det_x, (int)__det_y);
			}

		}

		// download
		download_volume_from_gpu(__model_2, (int)__qmax_len, 2);
		download_volume_from_gpu(__merge_w, (int)__qmax_len, 0);
		// all reduce (+) __model_2 and __merge_w
		MPI_Allreduce(MPI_IN_PLACE, __model_2, (int)__qmax_len*__qmax_len*__qmax_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, __merge_w, (int)__qmax_len*__qmax_len*__qmax_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		// merge scaling
		upload_models_to_gpu(NULL, __model_2, __merge_w, (int)__qmax_len);
		merge_scaling(BlockSize, BlockSize, 1);
		download_volume_from_gpu(__model_2, (int)__qmax_len, 2);

		// update scaling factor
		if(__scale){
			MPI_Allreduce(MPI_IN_PLACE, scaling_update_denominator, (int)__num_data, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
			memcpy_device_scaling_buf(scaling_update_denominator, (int)__num_data);
			update_scaling((int)__num_data);
		}


		thisp = NULL;
		time(&loop_end);
		diff_t = difftime(loop_end, loop_start);



		/* evaulate changes & rescale __model_2 */
		if( __MPIRANK == 0 )
			printf("\t* Result evaulation ...\n");

		diff_model = 0;
		model_norm = 0;
		rescale_model_2 = 0;

		if(__scale){

#pragma omp parallel for schedule(static,1) reduction(+:model_norm,diff_model)
			for(i=0; i<__qmax_len*__qmax_len*__qmax_len; i++){

				model_norm += pow((double)__model_1[i], 2.0);
				diff_model += pow((double)__model_2[i], 2.0);

			}
#pragma omp barrier

			rescale_model_2 = sqrt(model_norm / diff_model);
			diff_model = 0;

#pragma omp parallel for schedule(static,1) reduction(+:model_norm,diff_model)
			for(i=0; i<__qmax_len*__qmax_len*__qmax_len; i++){

				__model_2[i] *= (float)rescale_model_2;
				diff_model += pow((double)__model_2[i]-__model_1[i], 2.0);

			}
#pragma omp barrier

		}

		else{

#pragma omp parallel for schedule(static,1) reduction(+:model_norm,diff_model)
			for(i=0; i<__qmax_len*__qmax_len*__qmax_len; i++){

				model_norm += pow((double)__model_1[i], 2.0);
				diff_model += pow((double)__model_2[i]-__model_1[i], 2.0);

			}
#pragma omp barrier

		}

		diff_model = sqrt(diff_model/model_norm);

		if( __MPIRANK == 0 )
			printf("\t\trelative rmsd change : %lf\n", diff_model);

		if(__scale && __MPIRANK == 0)
			printf("\t\tintensity rescale    : %f\n", rescale_model_2);

		/* evaulate KL divergence of probabilities  */
		KL_entropy /= __num_data;

		if( __MPIRANK == 0 )
			printf("\t\tprobs KL-divergence  : %lf\n", KL_entropy);


		/* write log & update beta  */

		if( __MPIRANK == 0 ){
			// write log
			write_log(__iter_now, (float)diff_t, (float)diff_model, __beta, __quat_num, KL_entropy, PROB_MIN);

			// save info
			sprintf(line, "%s/info/info_%.3u.txt", __output_dir, __iter_now);
			fp = fopen(line, "w");
			if(__scale)
				fprintf(fp, "%f, %lf\n", rescale_model_2, KL_entropy);
			else
				fprintf(fp, "1.0, %lf\n", KL_entropy);
			fclose(fp);

			// save model_2
			sprintf(line, "%s/model/model_%.3u.bin", __output_dir, __iter_now);
			fp = fopen(line, "w");
			fwrite(__model_2, sizeof(float), __qmax_len*__qmax_len*__qmax_len, fp);
			fclose(fp);

			// save merge_w
			sprintf(line, "%s/merge_w/weight_%.3u.bin", __output_dir, __iter_now);
			fp = fopen(line, "w");
			fwrite(__merge_w, sizeof(float), __qmax_len*__qmax_len*__qmax_len, fp);
			fclose(fp);

			// save probabilities
			if(__iter_now % 5 == 0 || __iter_now == __iter_all || __iter_now == 1){
				rdata *rsort = (rdata*)malloc(__quat_num * sizeof(rdata));
				rdata buffer;
				sprintf(line, "%s/probs/probs_%.3u_top10.dat", __output_dir, __iter_now);
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

			// save scaling factor
			if(__scale){
				download_scaling_from_gpu(__scaling_factor, __num_data);
				sprintf(line, "%s/scaling/scaling_%.3u.dat", __output_dir, __iter_now);
				fp = fopen(line, "w");
				for(j=0; j<__num_data; j++){
					fprintf(fp, "%.4f\n", __scaling_factor[j]);
				}
				fclose(fp);
			}

		}

		// __iter_now start from 1
		if(__iter_now % __beta_jump == 0){
			__beta *= __beta_mul;
		}

		/* update __model_1 & __model_2 & __merge_w  */
		upload_models_to_gpu(__model_2, NULL, NULL, (int)__qmax_len);  // update __model_1_gpu
		reset_model_gpu((int)__qmax_len, 2);  // reset __model_2_gpu
		reset_model_gpu((int)__qmax_len, 0);  // reset __w_gpu
		// copy __model_2 to  __model_1
		memcpy(__model_1, __model_2, sizeof(__model_2));
		
		// refresh criterion
		KL_entropy = 0;
		// reset myslice=0 on gpu
		memcpy_device_slice_buf(NULL, (int)__det_x, (int)__det_y);
		// reset __P_jk to 0 (because MPI reduce them in different ranks)
		for(i=0; i<__num_data*__quat_num; i++) __P_jk = 0;
		// reset slice_intensity_r
		for(i=0; i<__quat_num; i++) slice_intensity_r = 0;
		// reset scaling_update_denominator
		if(__scale){
			for(i=0; i<__num_data; i++) scaling_update_denominator = 0;
		}
		
	}


	/*
	free
	*/
	free_device_all();
	free_host_global();
	// free host local var
	if(__scale){
		free(slice_intensity_r);
		free(scaling_update_denominator);
	}

	if( __MPIRANK == 0 ){
		time(&program_end);
		diff_t = difftime(program_end, program_start);
		printf("Total used time : %lf s\n", -diff_t);
	}

	MPI_Finalize();
}
