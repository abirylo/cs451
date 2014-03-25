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
#define FILE_BUFF 1024*1024*50	//copy 50MB of the file at a time

__host__ __device__ void iterate(unsigned int* table);
__host__ __device__ unsigned int get(unsigned int* table, unsigned short key);

__device__ void put(unsigned int* table, unsigned short key){
        atomicAdd(&table[(int)key], 1);
}

__host__ __device__ unsigned int get(unsigned int* table, unsigned short key){
  	unsigned int ret = table[(int)key];//(unsigned long)location2->value;
  	return ret;
}

__global__ void add_to_table( unsigned short *keys, unsigned int* table, int num_keys ) {
  	// get the thread id
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  	int stride = blockDim.x * gridDim.x; // total num of threads.

  	while (tid < num_keys) {//stripe
		unsigned short key = keys[tid];
		//printf("add_to_table: key = %u, tid = %d, table[tid] = %u\n", key, tid, table[tid]);
    		if(key != 0) put(table, key);
    		tid += stride;
  }
	__syncthreads();
}

void output_table(unsigned int* table){
  	FILE *outFile;
	outFile = fopen("GPUoutput.log", "w");

	for(int i=1; i<ELEMENTS; i++){
		fprintf(outFile, "key = %d ",i);
		fprintf(outFile, "value = %u}\n",table[i]);
  	}
	fclose(outFile);
}

void test_buffer(unsigned short* buffer, size_t buffer_size){
	unsigned short value;
	for(int i=0; i<buffer_size/2; i++){
		value = buffer[i];
		printf("value = %u\n", value);
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

	printf("File Opened.\n");

	unsigned short *buffer = (unsigned short*)calloc(1, FILE_BUFF);

  // allocate memory on the device for keys
	HANDLE_ERROR( cudaMalloc( (void**)&dev_buff, FILE_BUFF ) );

  // Initialize table on device
	HANDLE_ERROR( cudaMalloc( (void**)&dev_table, ELEMENTS * sizeof(unsigned int) ) );
	HANDLE_ERROR( cudaMemset( dev_table, 0, ELEMENTS * sizeof(unsigned int) ) );

  	printf("Calling GPU func with: ");
	size_read = fread(buffer, 1, FILE_BUFF, fd);
	int num_keys = size_read/2; //2 bytes per each key
	printf("%d threads, and a buffer of size %u (%d keys)\n", numThreads, size_read, num_keys);

	while(size_read != 0)
	{
		//printf("Attempting to copy %u bytes to the GPU.", size_read);
  		HANDLE_ERROR( cudaMemcpy( dev_buff, buffer, size_read, cudaMemcpyHostToDevice ) );  //copy chunk of data to device
  		add_to_table<<<32,numThreads>>>( dev_buff, dev_table, num_keys );
		cudaDeviceSynchronize();
		size_read = fread(buffer, 1, FILE_BUFF, fd);
  		num_keys = size_read/2;
	}
	fclose(fd);

  	unsigned int *table = (unsigned int*)calloc(ELEMENTS, sizeof(unsigned int));
	HANDLE_ERROR( cudaMemcpy( table, dev_table, ELEMENTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	output_table(table);

	HANDLE_ERROR( cudaFree( dev_table ) );
  	HANDLE_ERROR( cudaFree( dev_buff ) );
	free(table);
  	free( buffer );
  	return 0;
}
