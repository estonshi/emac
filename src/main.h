#ifndef EMAC_MAIN
#define EMAC_MAIN


#include "params.h"
#include "cuda_funcs.h" 
#include <omp.h>


// local variables
int BlockSize;
int num_threads;
float PROB_MIN;
float total_p;
float prob_tmp;
float mean_count, total_mean_count;
float rescale_model_2;
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
float *pattern;  // need to free

float *tmp_model;

// output orientation probabilities
typedef struct rindex{
	float data;
	int index;
}rdata;

static int cmp(const void *a, const void *b){
	return (*(rdata*)b).data-(*(rdata*)a).data > 0 ? 1:-1;
}


#endif