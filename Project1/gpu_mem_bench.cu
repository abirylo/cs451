//gpu_bench.cu

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define CHECK_ERR(x)                                    \
  if (x != cudaSuccess) {                               \
    fprintf(stderr,"%s in %s at line %d\n",             \
	    cudaGetErrorString(err),__FILE__,__LINE__);	\
    exit(-1);						\
  }                                                     \


unsigned long MAX_OPS = 20000000;
const long MEGABYTE = 1048576;

__global__ void gpu_iops(unsigned long max_ops) {

//  int a = blockDim.x * blockIdx.x + threadIdx.x;
    
}


int main(int argc, char *argv[]) {
  
    char c;  
    char test = 'B';
    char rw = 'R';
    while ( (c = getopt(argc, argv, "r:t:") ) != -1)
    {
        switch (c) 
        {
            case 'r':
                rw = optarg[0];
                break;
            case 't':
                test = optarg[0];
                break;
            default:
                printf("Usage: ./benchCPU -n [number of threads]\n");
                return -1;
        }
    }
    struct timeval tv;
    long long start, stop;
    double secs;

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
    unsigned char *d_mem_pointer;
    unsigned char *mem_pointer;
    if(test == 'B')
    {
      
      err = cudaMalloc((void **) &d_mem_pointer, sizeof(unsigned char)*MEGABYTE);
      CHECK_ERR(err);
      mem_pointer = (unsigned char *)malloc(sizeof(unsigned char)*1);
      gettimeofday(&tv, NULL);
      start = tv.tv_sec*1000000LL + tv.tv_usec;
    
      for(unsigned long i = 0; i<MEGABYTE; i++)
      {
        err = cudaMemcpy((void *)&d_mem_pointer[i], (void *)mem_pointer, 1, cudaMemcpyHostToDevice);
        CHECK_ERR(err);
      }
      
      gettimeofday(&tv, NULL);
      stop = tv.tv_sec*1000000LL + tv.tv_usec;
      secs = (stop-start)/1000000.0;
      printf("Time taken: %lf\n", secs);
      printf("%lf MB/sec\n", 1.0/(secs)); 
    }
    else if(test == 'K')
    {
      err = cudaMalloc((void **) &d_mem_pointer, sizeof(unsigned char)*256*MEGABYTE);
      CHECK_ERR(err);
      mem_pointer = (unsigned char *)malloc(sizeof(unsigned char)*1024);
      gettimeofday(&tv, NULL);
      start = tv.tv_sec*1000000LL + tv.tv_usec;
    
      for(unsigned long i = 0; i<256*MEGABYTE/1024; i++)
      {
        err = cudaMemcpy((void *)&d_mem_pointer[i*1024], (void *)mem_pointer, 1024, cudaMemcpyHostToDevice);
        CHECK_ERR(err);
      }
      
      gettimeofday(&tv, NULL);
      stop = tv.tv_sec*1000000LL + tv.tv_usec;
      secs = (stop-start)/1000000.0;
      printf("Time taken: %lf\n", secs);
      printf("%lf MB/sec\n", (256.0/1024.0)/(secs)); 
    }
    else if(test == 'M')
    {
      err = cudaMalloc((void **) &d_mem_pointer, sizeof(unsigned char)*512*MEGABYTE);
      CHECK_ERR(err);
      mem_pointer = (unsigned char *)malloc(sizeof(unsigned char)*MEGABYTE);
      gettimeofday(&tv, NULL);
      start = tv.tv_sec*1000000LL + tv.tv_usec;
    
      for(unsigned long i = 0; i<512*10; i++)
      {
        err = cudaMemcpy((void *)&d_mem_pointer[(i*MEGABYTE)%(512*MEGABYTE)], (void *)mem_pointer, MEGABYTE, cudaMemcpyHostToDevice);
        CHECK_ERR(err);
      }
      
      gettimeofday(&tv, NULL);
      stop = tv.tv_sec*1000000LL + tv.tv_usec;
      secs = (stop-start)/1000000.0;
      printf("Time taken: %lf\n", secs);
      printf("%lf MB/sec\n", (512*10)/(secs)); 
    }
    /*err = cudaFree(d_contents);
    CHECK_ERR(err);
    err = cudaFree(d_string_array);
    CHECK_ERR(err);
    err = cudaFree(d_answer_array);
    CHECK_ERR(err);
 */
}
