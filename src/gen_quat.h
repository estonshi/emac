#ifndef EMAC_GEN_QUAT
#define EMAC_GEN_QUAT

#include "base.h"

void cross_mul(double* v1, double* v2, double* v3, double norm);
void quat_mul(double* q1, double* q2, double* q3);
void quat_rot(double* q, double* v, double* v1);
double dot_mul(double* v1, double* v2);
void quat2rot(double q[4], double rot[3][3]);
void make_quat_rot(double q[4], double v[3], double v1[3], double shift);

int cal_quat_num(int num_level);
void gen_quaternions(int num_level, int mode, float* quaternions);

#endif