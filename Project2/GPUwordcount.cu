/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include "book.h"

#define ELEMENTS 65536			//Default 65536 for your dataset
//#define SIZE ELEMENTS*sizeof(unsigned int)
#define FILE_BUFF 1024*1024*10	//copy 1MB of the file at a time

__host__ __device__ void iterate(unsigned int* table);
__host__ __device__ unsigned int get(unsigned int* table, unsigned short key);

__device__ void put(unsigned int* table, unsigned short key){
        atomicAdd(&table[key], 1);
}

__host__ __device__ unsigned int get(unsigned int* table, unsigned short key){
  	unsigned int ret = table[key];//(unsigned long)location2->value;
  	return ret;
}

__global__ void add_to_table( unsigned short *keys, unsigned int* table ) {
  	// get the thread id
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  	int stride = blockDim.x * gridDim.x;// total num of threads.

  	while (tid < ELEMENTS) {//stripe
		unsigned short key = keys[tid];
		printf("add_to_table: key = %u, tid = %d, table[tid] = %u\n", key, tid, table[tid]);
    		put(table, key);
    		tid += stride;
  }
	__syncthreads();
}

// copy table back to host, verify elements are there
void verify_table( const unsigned int* dev_table ) {
	unsigned int* host_table;
	host_table = (unsigned int*)calloc(ELEMENTS, sizeof(unsigned int));
	HANDLE_ERROR( cudaMemcpy( host_table, dev_table, ELEMENTS*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
  	iterate(host_table);
  	printf("END VERIFY TABLE\n");
}

__host__ __device__ void iterate(unsigned int* table){
  	for(int i=1; i<ELEMENTS; i++){
    		printf("[%d]: {", i);
		unsigned key = i;
		printf("key = %u ",key);
		printf("value = %u}\n",table[key]);
  	}
}

int main(int argc, char *argv[]) {
  	printf("Starting main.\n");
  	printf("Elements = %u\n", ELEMENTS);
	int numThreads = atoi(argv[1])/32;
	size_t size_read;

  	unsigned short *dev_buff;
	unsigned int *dev_table;

	FILE *fd;
	fd = fopen(argv[2], "rb");
	if (fd == NULL){
		fputs ("File error",stderr);
		exit (1);
	}

	unsigned short *buffer = (unsigned short*)calloc(1, FILE_BUFF);

  // allocate memory on the device for keys and copy to device
	HANDLE_ERROR( cudaMalloc( (void**)&dev_buff, FILE_BUFF ) );

  // Initialize table on device
	HANDLE_ERROR( cudaMalloc( (void**)&dev_table, ELEMENTS * sizeof(unsigned int) ) );
	HANDLE_ERROR( cudaMemset( dev_table, 0, ELEMENTS * sizeof(unsigned int) ) );

  	printf("Calling GPU func\n");
	while(!feof(fd)){
		size_read = fread(buffer, 1, FILE_BUFF, fd);
  		HANDLE_ERROR( cudaMemcpy( dev_buff, buffer, size_read, cudaMemcpyHostToDevice ) );
  		add_to_table<<<1,numThreads>>>( dev_buff, dev_table);
  	}

	cudaDeviceSynchronize();

  	verify_table( dev_table );

  	HANDLE_ERROR( cudaFree( dev_buff ) );
  	free( buffer );
  	return 0;
}
