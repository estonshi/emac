#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main(){
    cudaError_t custatus;
    int gpu_num = 0;
    cudaDeviceProp prop;
    
    custatus = cudaGetDeviceCount(&gpu_num);
    printf("Number of GPUs       : %d\n", gpu_num);

    custatus = cudaSetDevice(0);
    if(custatus != cudaSuccess){
        printf("Failed to set Device 0. Exit\n");
        return -1;
    }

    cudaGetDeviceProperties(&prop, 0);
    printf("Device name          : %s\n", prop.name);
    printf("Compute Capability   : %d.%d\n", prop.major, prop.minor);
    printf("totalGlobalMem       : %.1f GB\n", prop.totalGlobalMem/1024/1024/1024.0);
    printf("sharedMemPerBlock    : %u KB\n", (unsigned int)prop.sharedMemPerBlock/1024);
    printf("maxThreadsPerBlock   : %d\n", prop.maxThreadsPerBlock);
    printf("maxGridSize          : %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("regPerBlock          : %d\n", prop.regsPerBlock);
    
}
