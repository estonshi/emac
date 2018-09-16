#include "params.h"

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
