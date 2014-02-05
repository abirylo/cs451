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

__global__ void gpu_iops(unsigned long max_ops) {

//  int a = blockDim.x * blockIdx.x + threadIdx.x;
    
    int a=0;
    int b=0;
    int c=0;
    int d=0;
    int e=0;
    int f=0;
    int g=0;
    int h=0;
    int i=0;
    int j=0;
    int k=0;
    int l=0;
    int m=0;
    int n=0;
    int o=0;
    int p=0;
    int q=0;
    int r=0;
    int s=0;
    int t=0;
    int u=0;
    int v=0;
    int w=0;
    int x=0;
    
    for(unsigned long count=0; count<max_ops; count++)
    {
        a=a+1;
        b=b+2;
        c=c+3;
        d=d+4;
        e=e+5;
        f=f+6;
        g=g+7;
        h=h+8;
        i=i+9;
        j=j+10;
        k=k+11;
        l=l+12;
        m=m*13;
        n=n*14;
        o=o*15;
        p=p*16;
        q=q*17;
        r=r*18;
        s=s*19;
        t=t*20;
        u=u*21;
        v=v*22;
        w=w*23;
        x=x*24;
    }
}

__global__ void gpu_flops(unsigned long max_ops) {

//    int a = blockDim.x * blockIdx.x + threadIdx.x;
    float a=0.0;
    float b=0.0;
    float c=0.0;
    float d=0.0;
    float e=0.0;
    float f=0.0;
    float g=0.0;
    float h=0.0;
    float i=0.0;
    float j=0.0;
    float k=0.0;
    float l=0.0;
    float m=0.0;
    float n=0.0;
    float o=0.0;
    float p=0.0;
    float q=0.0;
    float r=0.0;
    float s=0.0;
    float t=0.0;
    float u=0.0;
    float v=0.0;
    float w=0.0;
    float x=0.0;
    for(unsigned long count=0; count<max_ops; count++)
    {
        a=a+1.1;
        b=b+2.2;
        c=c+3.3;
        d=d+4.4;
        e=e+5.5;
        f=f+6.6;
        g=g+7.7;
        h=h+8.8;
        i=i+9.9;
        j=j+10.10;
        k=k+11.11;
        l=l+12.12;
        m=m*13.13;
        n=n*14.14;
        o=o*15.15;
        p=p*16.16;
        q=q*17.17;
        r=r*18.18;
        s=s*19.19;
        t=t*20.20;
        u=u*21.21;
        v=v*22.22;
        w=w*23.23;
        x=x*24.24;
    }
}

int main(int argc, char *argv[]) {
  
    char c;  
    int threads = 1024;
    char test = 'I';
    while ( (c = getopt(argc, argv, "n:l:t:") ) != -1) 
    {
        switch (c) 
        {
            case 'n':
                threads = atoi(optarg);
                break;
            case 'l':
                MAX_OPS = atol(optarg);
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

    if(test == 'I')
    {
      gettimeofday(&tv, NULL);
      start = tv.tv_sec*1000000LL + tv.tv_usec;
    
      gpu_iops<<< ceil(threads/1024), 1024 >>>(MAX_OPS);        
      cudaThreadSynchronize();
 
      gettimeofday(&tv, NULL);
      stop = tv.tv_sec*1000000LL + tv.tv_usec;
      secs = (stop-start)/1000000.0;
      //printf("Time taken: %lf\n", secs);
      printf("I\t%lf\n", (MAX_OPS*24.*threads)/(secs*1000000000.)); 
    }
    else if(test == 'F')
    {
      gettimeofday(&tv, NULL);
      start = tv.tv_sec*1000000LL + tv.tv_usec;
      
      gpu_flops<<< ceil(threads/1024), 1024 >>>(MAX_OPS);        
      cudaThreadSynchronize();
 
      gettimeofday(&tv, NULL);
      stop = tv.tv_sec*1000000LL + tv.tv_usec;
      secs = (stop-start)/1000000.0;
      //printf("Time taken: %lf\n", secs);
      printf("FL\t%lf\n", (MAX_OPS*24.*threads)/(secs*1000000000.)); 
    }
}
