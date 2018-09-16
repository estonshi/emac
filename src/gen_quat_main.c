#include "gen_quat.h"


int main(int argc, char** argv){
	int num_level = 30;
	char save_file[999];
	strcpy(save_file, "../input/orientations.quat");

	int i, c;
	bool change_param = false;
	int mode = 0;
	while( (c = getopt(argc, argv, "n:s:m:h")) != -1 ){
		switch (c){
			case 'n':
				num_level = atoi(optarg);
				change_param = true;
				break;
			case 's':
				strcpy(save_file, optarg);
				change_param = true;
				break;
			case 'm':
				mode = atoi(optarg);
				change_param = true;
				if (mode<0 || mode>1){
					printf("Error : There are only 2 modes, =0 or =1\n");
					return 1;
				}
				break;
			case 'h':
				printf("\nThis function is used to generate uniformly distributed orientations.");
				printf("\nOptions : ");
				printf("\n          -n [number_level] ");
				printf("\n          -s [save file] ");
				printf("\n          -m [mode] ( 0/1, 0 : uniform distribution, 1 : random distribution )\n\n");
				return 0;
				break;
			case '?':
				printf("\nUnknown option character '-%c'", optopt);
				return 1;
				break;
			default:
				printf("\nDo nothing. Exit.");
				return 0;
		}
	}

	if(!change_param){
		char reply[2];
		char yes[] = "y";
		printf("Use default parameters : num_level=30, uniform 0 mode, save_path='../input/orientations.quat' ? (y/n)\n");
		scanf("%s", reply);
		if(strcmp(reply, yes) == 0) change_param = true;
	}

	if(!change_param) return 0;

	// calculate quaternion number
	int number = cal_quat_num(num_level);
	float* quaternions = (float*) malloc(number*4*sizeof(float)); 
	// calculate quaternions
	gen_quaternions(num_level, mode, quaternions);

	FILE *fp;
	fp = fopen(save_file, "w+");
	
	for(i=0; i<number; i++){
		fprintf(fp, "%.4f %.4f %.4f %.4f\n", quaternions[i*4], quaternions[i*4+1], quaternions[i*4+2], quaternions[i*4+3]);
	}

	fclose(fp);
	free((void *)quaternions);

}