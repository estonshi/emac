#include "emac_data.h"

bool load_emac_prop(char* filename, uint32* num_data, uint32* size_x, uint32* size_y){
	FILE* fp;
	fp = fopen(filename, "rb");
	if( fp == NULL ){
		printf("[error] Your data file is invalid.\n");
		return false;
	}
	fread(num_data, sizeof(uint32), 1, fp);
	fread(size_x, sizeof(uint32), 1, fp);
	fread(size_y, sizeof(uint32), 1, fp);
	fclose(fp);
	return true;
}


float load_emac_dataset(char* filename, emac_pat* dataset){
	float mean_count = 0, photon_count;
	uint32 num_data, size_x, size_y;
	int i, k;
	FILE* fp;
	fp = fopen(filename, "rb");
	if( fp == NULL ){
		printf("[error] Your data file is invalid.\n");
		return -1;
	}
	fread(&num_data, sizeof(uint32), 1, fp);
	fread(&size_x, sizeof(uint32), 1, fp);
	fread(&size_y, sizeof(uint32), 1, fp);
	// read data
	emac_pat* thisp = dataset;
	uint32 one_count, mul_count, data_len;
	for(i=0; i<num_data; i++){
		fread(&data_len, sizeof(uint32), 1, fp);
		fread(&one_count, sizeof(uint32), 1, fp);
		mul_count = (data_len - one_count - 2)/2;
		thisp->one_pix = one_count;
		thisp->mul_pix = mul_count;
		thisp->one_loc = malloc(one_count * sizeof(uint32));
		fread(thisp->one_loc, sizeof(uint32), one_count, fp);
		thisp->mul_loc = malloc(mul_count * sizeof(uint32));
		fread(thisp->mul_loc, sizeof(uint32), mul_count, fp);
		thisp->mul_counts = malloc(mul_count * sizeof(uint32));
		fread(thisp->mul_counts, sizeof(uint32), mul_count, fp);
		if(i < num_data - 1)
			thisp->next = (emac_pat*) malloc(sizeof(emac_pat));
		else
			thisp->next = NULL;
		// calculate total photon count
		photon_count = (float)one_count;
		for(k=0; k<mul_count; k++)
			photon_count += (float)thisp->mul_counts[k];
		thisp->photon_count = (float)photon_count;
		mean_count += photon_count/(float)num_data;
		// next pattern
		thisp = thisp->next;
	}
	// update scale factor
	thisp = dataset;
	while(thisp != NULL){
		thisp->scale_factor = mean_count / thisp->photon_count;
		thisp = thisp->next;
	}
	fclose(fp);
	return mean_count;
}


float parse_pat_full(emac_pat* pat_struct, uint32 size_x, bool scale, float** this_pat){
	uint32 one_count = pat_struct->one_pix;
	uint32 mul_count = pat_struct->mul_pix;
	uint32 x, y;
	int i;
	float scale_factor = 1.0;
	if(scale) scale_factor = pat_struct->scale_factor;
	// one photon
	for(i=0; i<one_count; i++){
		uint32 loc = pat_struct->one_loc[i];
		x = loc/size_x;
		y = loc%size_x;
		this_pat[x][y] = 1.0 * scale_factor;
	}
	// multi photon
	for(i=0; i<mul_count; i++){
		uint32 loc = pat_struct->mul_loc[i];
		uint32 count = pat_struct->mul_counts[i];
		x = loc/size_x;
		y = loc%size_x;
		this_pat[x][y] = (float)count * scale_factor;
	}
	float photons = pat_struct->photon_count * scale_factor;
	return photons;
}

