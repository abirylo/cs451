/*
 * Adrian Birylo
 * abirylo@iit.edu
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>


#define CHECK_ERR(x)                                    \
  if (x != cudaSuccess) {                               \
    fprintf(stderr,"%s in %s at line %d\n",             \
	    cudaGetErrorString(err),__FILE__,__LINE__);	\
    exit(-1);						\
  }                                                     \

__global__ void gpu() {

  int a = blockDim.x * blockIdx.x + threadIdx.x;

  if(a <= n)
  {
    d_out[a] = false;
    bool match = false;
    int i=0;
    int j=0;
  }
}

int main(int argc, char *argv[]) {
  
    cudaError_t err;
   
/*  
    err = cudaMalloc((void **) &d_string_array, sizeof(char**)*array_size); 
    CHECK_ERR(err); 

    err = cudaMalloc((void **) &d_answer_array, sizeof(bool)*array_size);
    CHECK_ERR(err);
      
    d_answer_array_copy = d_answer_array;

    err = cudaMalloc((void **) &d_command, line_size);
    CHECK_ERR(err);  

    err = cudaMalloc((void **) &d_contents, array_size*line_size);
    CHECK_ERR(err);
      
    err = cudaMemcpy(d_command, argv[1], strlen(argv[1])+1, cudaMemcpyHostToDevice);
    CHECK_ERR(err);      
*/
    CUdevprop prop = malloc(sizeof(CUdevprop));
             //run grep    
    gpu<<< ceil(array_size/1024), 1024 >>>();        
 

 /*
    err = cudaFree(d_contents);
    CHECK_ERR(err);
    err = cudaFree(d_string_array);
    CHECK_ERR(err);
    err = cudaFree(d_answer_array);
    CHECK_ERR(err);
 */
  }
}
