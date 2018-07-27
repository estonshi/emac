#include "gen_quat.h"

// v3 = v1 x v2, right hand rule
void cross_mul(double* v1, double* v2, double*v3, double norm){
	v3[0] = v1[1]*v2[2] - v1[2]*v2[1];
	v3[1] = v1[2]*v2[0] - v1[0]*v2[2];
	v3[2] = v1[0]*v2[1] - v1[1]*v2[0];
	if(norm){
		double scale = 1.0/sqrt(v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2]);
		v3[0] *= scale;
		v3[1] *= scale;
		v3[2] *= scale;
	}
}

// v3 = v1 . v2
double dot_mul(double* v1, double* v2){
	return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

// q3 = q1 * q2
void quat_mul(double* q1, double* q2, double* q3){
	q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
	q3[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[3]*q2[2] - q1[2]*q2[3];
	q3[2] = q1[0]*q2[2] + q1[2]*q2[0] + q1[1]*q2[3] - q1[3]*q2[1];
	q3[3] = q1[0]*q2[3] + q1[3]*q2[0] + q1[2]*q2[1] - q1[1]*q2[2];
}

// v1 = q*v*q^(-1)
void quat_rot(double* q, double* v, double* v1){
	double q0 = q[0];
	double qn[3] = {q[1], q[2], q[3]};
	double q_len = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
	cross_mul(qn, v, v1, 1);
	v1[0] = (q0*q0 - q_len)*v[0] + 2.0*dot_mul(q, v)*q[0] + 2.0*q0*v1[0];
	v1[1] = (q0*q0 - q_len)*v[1] + 2.0*dot_mul(q, v)*q[1] + 2.0*q0*v1[1];
	v1[2] = (q0*q0 - q_len)*v[2] + 2.0*dot_mul(q, v)*q[2] + 2.0*q0*v1[2];
}

// quaternion to rotation matrix
void quat2rot(double q[4], double rot[3][3]){
	double q0, q1, q2, q3, q01, q02, q03, q11, q12, q13, q22, q23, q33;
	
	q0 = q[0];
	q1 = q[1];
	q2 = q[2];
	q3 = q[3];
	
	q01 = q0*q1;
	q02 = q0*q2;
	q03 = q0*q3;
	q11 = q1*q1;
	q12 = q1*q2;
	q13 = q1*q3;
	q22 = q2*q2;
	q23 = q2*q3;
	q33 = q3*q3;
	
	rot[0][0] = (1. - 2.*(q22 + q33));
	rot[0][1] = 2.*(q12 + q03);
	rot[0][2] = 2.*(q13 - q02);
	rot[1][0] = 2.*(q12 - q03);
	rot[1][1] = (1. - 2.*(q11 + q33));
	rot[1][2] = 2.*(q01 + q23);
	rot[2][0] = 2.*(q02 + q13);
	rot[2][1] = 2.*(q23 - q01);
	rot[2][2] = (1. - 2.*(q11 + q22));
}


// rotate 
void make_quat_rot(double q[4], double v[3], double v1[3], double shift) {
	double rot[3][3];
	quat2rot(q, rot);

	for(int i=0; i<3; i++){
		v1[i] = 0;
		for(int j=0; j<3; j++){
			v1[i] += rot[i][j] * v[j];
		}
		v1[i] += shift;
	}
}


// calculate the number of quaternions
int cal_quat_num(int num_level){
	return (int)(4.0*num_level*num_level/PI/PI) * num_level;
}


// generate quaternions
void gen_quaternions(int num_level, int mode, float* quaternions){
	// init
	int N = (int)(4.0*num_level*num_level/PI/PI);
	double phi;
	double point[3] = {0,0,0};

	// calculate orientations and write to file
	int ii;
	double theta, cphi, sphi;
	double ang_1;
	double axis_1[3] = {0,0,0};
	double quat_final[4];
	double init[3] = {0,0,1};

	for(int i=0;i<N;i++){
		switch (mode){
			// two different modes
			case 0:
				ii = 2*i-(N-1);
				theta = 2.0*PI*ii/FAB;
				sphi = (double)ii/(double)N;
				cphi = sqrt((double)(N+ii)*(N-ii))/(double)N;
				point[0] = cphi*sin(theta);
				point[1] = cphi*cos(theta);
				point[2] = sphi;
				break;
			case 1:
				sphi = 1.0/(double)N;
				srandom(i);
				ii = random() % (5*N);
				theta = 2.0 * PI * ( sphi * ii / 5.0 );
				ii = random() % (5*N);
				cphi = 1.0 - 1e-5 - ( 2.0 - 2e-5 ) * ( sphi * ii / 5.0 );
				point[0] = sqrt(1 - cphi*cphi) * cos(theta);
				point[1] = sqrt(1 - cphi*cphi) * sin(theta);
				point[2] = cphi;
				break;
			default:
				return;
				break;
		}
		// calculate rotation angle and axis
		ang_1 = -acos(dot_mul(init, point));
		cross_mul(init, point, axis_1, 1);
		double quat_1[4] = {cos(ang_1/2.0), axis_1[0]*sin(ang_1/2.0), axis_1[1]*sin(ang_1/2.0), axis_1[2]*sin(ang_1/2.0)};
		// calculate inner rotation
		for(int j=0;j<num_level;j++){
			phi = j*2.0*PI/num_level;
			double quat_2[4] = {cos(phi/2.0), point[0]*sin(phi/2.0), point[1]*sin(phi/2.0), point[2]*sin(phi/2.0)};
			quat_mul(quat_2, quat_1, quat_final);
			// (w, qx, qy, qz)
			quaternions[ 4*(i*num_level+j) ] = (float)quat_final[0];
			quaternions[ 4*(i*num_level+j)+1 ] = (float)quat_final[1];
			quaternions[ 4*(i*num_level+j)+2 ] = (float)quat_final[2];
			quaternions[ 4*(i*num_level+j)+3 ] = (float)quat_final[3];
		}
	}
}

