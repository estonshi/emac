#ifndef EMACS_CUDA_FUNCS
#define EMACS_CUDA_FUNCS


/*   gpu setup & init & utils   */

extern void setDevice(int gpu_id);

extern void gpu_var_init(int det_x, int det_y, float det_center[2], int num_mask_ron[2], 
	int vol_size, int stoprad, int quat_num, float *quaternion, float *ori_det, 
	int *ori_mask, float *init_model_1, float *init_model_2, float *init_merge_w, int ang_corr_bins);

extern void upload_models_to_gpu(float *model_1, float *model_2, float *merge_w, int vol_size);

extern void download_model2_from_gpu(float *model_2, int vol_size);

extern void download_volume_from_gpu(float *vol_container, int vol_size, int which);

extern void download_currSlice_from_gpu(float *new_slice, int det_x, int det_y);

extern void reset_model(int vol_size, int which);

extern void memcpy_device_pattern_buf(float *pattern, int det_x, int det_y);

extern void memcpy_device_slice_buf(float *myslice, int det_x, int det_y);

extern void free_cuda_all();

extern void cuda_start_event();

extern float cuda_return_time();


/*   slicing and merging   */

extern void get_slice(int quat_index, float *myslice, int BlockSize, int det_x, int det_y, int MASKPIX);

extern void merge_slice(int quat_index, float *myslice, int BlockSize, int det_x, int det_y);

extern void merge_scaling(int GridSize, int BlockSize, float scal_factor); 


/*   angular correlation   */

extern void do_angcorr(int partition, float* pat, float *result, int det_x, int det_y, int BlockSize, bool inputType);

extern float comp_angcorr(int partition, float *model_slice_ac, float *pattern_ac, int BlockSize);


/*       likelihood        */

extern float calc_likelihood(float beta, float *model_slice, float *pattern, int det_x, int det_y);

extern void maximization_dot(float *pattern, float prob, int det_x, int det_y, float *new_slice, int BlockSize);

extern void maximization_norm(float scaling_factor, int det_x, int det_y, int BlockSize);


#endif