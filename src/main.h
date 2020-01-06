#ifndef EMAC_MAIN
#define EMAC_MAIN


#include "params.h"
#include "cuda_funcs.h" 
#include <omp.h>
#include "mpi.h"

extern void gpu_dataset_init(emac_pat *dataset, int num_pat);

/* local variables */
int BlockSize;
int num_threads;
int threads_id;
float PROB_MIN;
float scale_tmp;
float mean_count;

double rescale_model_2;
double total_mean_count;
double total_p;
double prob_tmp;
double diff_model;
double model_norm;
double KL_entropy;
double Q_prob;

char config_file[999], line[999];
FILE *fp;
int i, j, c, gpu_id;
bool resume, succeed;

time_t loop_start, loop_end, program_start, program_end;
double diff_t;

emac_pat *thisp;
float *slice_intensity_r; // used for update scaling, need to free
float *scaling_update_denominator; // the denominator of scaling factor updated rule, need tp free

char *token;


/* global variables */
int __MPIRANK, __NUMPROC;


/* output orientation probabilities */
typedef struct rindex{
	float data;
	int index;
}rdata;

static int cmp(const void *a, const void *b){
	return (*(rdata*)b).data-(*(rdata*)a).data > 0 ? 1:-1;
}


#endif