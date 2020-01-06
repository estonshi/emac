#include "main.h"



int main(int argc, char** argv){

	//
	if( __MPIRANK == 0 )
		time(&program_start);
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &__MPIRANK);
	MPI_Comm_size(MPI_COMM_WORLD, &__NUMPROC);

	gpu_id = 9;
   	while( (c = getopt(argc, argv, "c:g:h")) != -1 ){
		switch (c){
			case 'c':
				strcpy(config_file, optarg);
				break;
			case 'g':
				token = strtok(optarg,",");
				i = 0;
				while(i != __MPIRANK){
					token = strtok(NULL, ",");
					i++;
					if(token == NULL){
						printf("[ERROR] The number of given GPU(s) is smaller than MPI size !");
						MPI_Finalize();
						exit(-1);
					}
				}
				gpu_id = atoi(token);
				break;
			default:
				printf("\nDo nothing. Exit.");
				return 0;
		}
	}
	resume = false;
	BlockSize = 16;
	num_threads = 1;
	__iter_all = 10;

	// openmp
	omp_set_num_threads(1);

	// host set up
	succeed = setup(config_file, resume, __MPIRANK);
	if(!succeed){
		free_host_global();
		MPI_Finalize();
		exit(-1);
	}

	if(__NUMPROC > 1 && resume == false){
                MPI_Bcast(__model_1, (int)__qmax_len*__qmax_len*__qmax_len, MPI_FLOAT, 0 , MPI_COMM_WORLD);
	}

	pattern = (float*) calloc((int)__det_x*__det_y, sizeof(float));
	if(__scale){
		slice_intensity_r = (float*) calloc((int)__num_data, sizeof(float));
		scaling_update_denominator = (float*) calloc((int)__num_data, sizeof(float));
	}

	Q_prob = 1.0/__quat_num;
	KL_entropy = 0;
	tmp_model = NULL;
	MPI_Barrier(MPI_COMM_WORLD);

	// GPU init
	printf("Copying files to GPU %d ...\n", gpu_id);
	setDevice(gpu_id);
	gpu_var_init((int)__det_x, (int)__det_y, __center_p, __num_mask_ron, (int)__qmax_len, (int)__ron, (int)__quat_num, __quat, __det, __correction, __mask, __model_1, __model_2, __merge_w, (int)__ang_corr_grid, num_threads);
	gpu_dataset_init(__dataset, (int)__num_data);
	printf("done.\n");
	free_dataset();

	MPI_Barrier(MPI_COMM_WORLD);

	for(; __iter_now<=__iter_all; __iter_now++)
	{
		
		if(__iter_now<=__iter_all/2)
			PROB_MIN = (2.0f*(float)(__iter_now-1)/(float)__iter_all)*(1.0f/__quat_num);
		else
			PROB_MIN = 1.0f/__quat_num;

		// read __P_jk
		if(__MPIRANK == 0){
			printf("read __P_jk\n");
			sprintf(line, "tst_parse-l.bin");
			fp = fopen(line, "r");
			fread(__P_jk, sizeof(double), (int)__num_data*(int)__quat_num, fp);
			fclose(fp);
			printf("read slice\n");
			sprintf(line, "./test/tst_merge_slice.bin");
			fp = fopen(line, "r");
			fread(pattern, sizeof(float), (int)__det_x*(int)__det_y, fp);
			fclose(fp);
		}
		MPI_Bcast(__P_jk, (int)__num_data*(int)__quat_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(pattern, (int)__det_x*(int)__det_y, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		printf("Start maximization\n");
		// maximization
		rescale_model_2 = 0;
		memcpy_device_slice_buf(pattern, (int)__det_x, (int)__det_y);
	
		for(i=0; i<__quat_num; i++)
		{
			if(i % __NUMPROC != __MPIRANK){
				continue;
			}
			
			// reset muslice to 0
			memcpy_device_slice_buf(NULL, (int)__det_x, (int)__det_y);
			total_p = 0;
			total_mean_count = 0;
			
			for(j=0; j<__num_data; j++){
				prob_tmp = __P_jk[i+j*__quat_num];
				
				if((float)prob_tmp < PROB_MIN) continue;

				//c = __photon_counts[j];
				//parse_pattern_ongpu(j, c, (int)__det_x, (int)__det_y, NULL, __scale);
				
				//maximization_dot(NULL, (float)prob_tmp, (int)__det_x, (int)__det_y, NULL, BlockSize);
				maximization_dot_part(j, (float)prob_tmp, __pix_hasphoton[j*3+1], __pix_hasphoton[j*3+2]);

				total_mean_count += (__dataset_mean_count * prob_tmp);
				total_p += prob_tmp;

			}
			total_p = 1.0;
			total_mean_count = 10000;
			
			//if(total_p > 0){
			//	rescale_model_2 += (float)(total_mean_count / total_p);
			//	maximization_norm((float)(1.0/total_p), (int)__det_x, (int)__det_y, BlockSize);
				merge_slice(i, NULL, BlockSize, (int)__det_x, (int)__det_y);
			//}
			/*
			if(i == 0 || i == 145 || i == 1234){
				download_currSlice_from_gpu(pattern, (int)__det_x, (int)__det_y );
				sprintf(line, "tst_merge_slice_%d.bin", i);
				fp = fopen(line, "w");
				fwrite(pattern, sizeof(float), (int)__det_x*(int)__det_y, fp);
				fclose(fp);
			}*/
			if(i%100==0) printf("processed %d\n", i);

		}

		//rescale_model_2 = __dataset_mean_count / (rescale_model_2 / (float)__quat_num);
		download_volume_from_gpu(__model_2, (int)__qmax_len, 2);
		download_volume_from_gpu(__merge_w, (int)__qmax_len, 0);
		MPI_Allreduce(MPI_IN_PLACE, __model_2, (int)__qmax_len*__qmax_len*__qmax_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, __merge_w, (int)__qmax_len*__qmax_len*__qmax_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		upload_models_to_gpu(NULL, __model_2, __merge_w, (int)__qmax_len);
		merge_scaling(BlockSize, BlockSize, 1);
		download_volume_from_gpu(__model_2, (int)__qmax_len, 2);
		printf("Done.\n");
		// write
		/*
		sprintf(line, "tst_merged_volume.bin");
		fp = fopen(line, "w");
		fwrite(__model_2, sizeof(float), (int)__qmax_len*__qmax_len*__qmax_len, fp);
		fclose(fp);
		sprintf(line, "tst_merged_weight.bin");
		fp = fopen(line, "w");
		fwrite(__merge_w, sizeof(float), (int)__qmax_len*__qmax_len*__qmax_len, fp);
		fclose(fp);
		*/

		exit(0);

	}
}
