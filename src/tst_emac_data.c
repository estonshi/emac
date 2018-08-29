#include "emac_data.h"


int main(int argc, char** argv){
	char input[999];
	char output[999];

	int i, j, c;
	while( (c = getopt(argc, argv, "i:o:h")) != -1 ){
		switch (c){
			case 'i':
				strcpy(input, optarg);
				break;
			case 'o':
				strcpy(output, optarg);
				break;
			case 'h':
				printf("\nThis program is used to test dataset parser.");
				printf("\nUsage : ./emac_data -i [input emac file] -o [output binary data file]\n\n");
				return 0;
				break;
			default:
				printf("\nThis program is used to test dataset parser.");
				printf("\nUsage : ./emac_data -i [input emac file] -o [output binary data file]\n\n");
				return 0;
				break;
		}
	}

	uint32 num_data, size_x, size_y;
	emac_pat* dataset = (emac_pat*) malloc(sizeof(emac_pat));
	load_emac_prop(input, &num_data, &size_x, &size_y);
	float mean_count = load_emac_dataset(input, dataset);

	FILE* fp;
	fp = fopen(output, "wb");
	emac_pat* this_data = dataset;
	// init pat
	float** pat = (float**) calloc(size_x, sizeof(float*));
	for(i=0; i<size_x; i++){
		pat[i] = (float*) calloc(size_y, sizeof(float));
	}
	float this_photons = 0;
	for(i=0; i<num_data; i++){
		if(i == 0 || i == num_data-1){
			this_photons = parse_pat_full(this_data, size_x, true, pat);
			for(j=0; j<size_x; j++){
				fwrite(pat[j], sizeof(float), size_y, fp);
			}
			printf("Total photons after scaling = %f\n", this_photons);
		}
		this_data = this_data->next;
	}
	// free pat
	for(i=0; i<size_y; i++){
		free((void *)pat[i]);
	}
	free((void *)pat);
	fclose(fp);

	return 0;
}