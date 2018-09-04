#ifndef EMAC_DATA
#define EMAC_DATA

#include "base.h"

typedef struct pattern
{
	uint32 one_pix, mul_pix;
	uint32* one_loc;
	uint32* mul_loc;
	uint32* mul_counts;
	float scale_factor;
	float photon_count;
	struct pattern* next;
} emac_pat;

// size_x is the number of rows of a pattern
bool load_emac_prop(char* filename, uint32* num_data, uint32* size_x, uint32* size_y);

float load_emac_dataset(char* filename, emac_pat* dataset);

float parse_pat_full(emac_pat* pat_struct, uint32 size_x, bool scale, float** this_pat); // deprecated

float parse_pattern(emac_pat* pat_struct, uint32 length, bool scale, float *this_pat);

#endif