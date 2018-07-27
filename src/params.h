#ifndef EMAC_PARAMS
#define EMAC_PARAMS

#include "base.h"
#include "emac_data.h"
#include "gen_quat.h"

char __data_path[999], __det_path[999], __quat_path[999], __output_dir[999], __log_file[999];
uint32 __quat_num, __det_x, __det_y;
float __center_x, __center_y;

bool __scale;
uint32 __num_data;
float __beta, __beta_mul;
uint32 __beta_jump;
uint32 __qmax_len;
uint32 __iter_now;
float __center_q[3];
float *__model_1, *__model_2, *__merge_w;
float *__det;
float *__quat; 
emac_pat* __dataset;
int* __mask;


bool load_det_info(char* det_file, int* mask, float* det);
bool load_model(char* model_file, uint32 size, float* model);
void write_log(uint32 iter_num, float used_time, float rms_change, float beta, uint32 num_rot, float pat_rot);
float read_log(char* read_item, int iter_num);

bool setup(char* config_file, bool resume);
void free_all();

#endif