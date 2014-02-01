//cpu_bench.c

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

unsigned long MAX_OPS = 20000000;

double timeDiff(struct timespec *start, struct timespec *end){
	return (double)(end->tv_sec-start->tv_sec)+((end->tv_nsec-start->tv_nsec)/1000000000.0);
}

void* cpuInt(void* arg)
{   
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
    
    for(unsigned long count=0; count<MAX_OPS; count++)
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
    //printf("%i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n", a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x);
    
    return NULL;
}

void* cpuFloat(void* arg)
{   
    float a=0;
    float b=0;
    float c=0;
    float d=0;
    float e=0;
    float f=0;
    float g=0;
    float h=0;
    float i=0;
    float j=0;
    float k=0;
    float l=0;
    float m=0;
    float n=0;
    float o=0;
    float p=0;
    float q=0;
    float r=0;
    float s=0;
    float t=0;
    float u=0;
    float v=0;
    float w=0;
    float x=0;
    for(unsigned long count=0; count<MAX_OPS; count++)
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
    //printf("%i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n", a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x);
    return NULL;
}

int main(int argc, char** argv)
{
    char c;
    int threads = 1;

    while ( (c = getopt(argc, argv, "n:l:") ) != -1) 
    {
        switch (c) 
        {
            case 'n':
                threads = atoi(optarg);
                break;
            case 'l':
                MAX_OPS = atol(optarg);
                break;
            default:
                printf("Usage: ./benchCPU -n [number of threads]\n");
                return -1;
                break;
        }
    }
    
//    clock_t start, stop;
//    struct timespec start, stop;
    struct timeval tv;
    long long start, stop;
    double secs;
    
    pthread_t cpu_threads[threads]; 
    
//    start = clock();
    gettimeofday(&tv, NULL);
    start = tv.tv_sec*1000000LL + tv.tv_usec;
    for(int i=0; i<threads; i++)
    {
       pthread_create(&cpu_threads[i], NULL, cpuInt, NULL);
    }
    
    for(int i=0; i<threads; i++)
    {
        pthread_join(cpu_threads[i], NULL);
    }
//    stop = clock();
//    clock_gettime(CLOCK_MONOTONIC, &stop);
//	secs = timeDiff(&start, &stop);
//    printf("Start Time: %lf, End Time: %lf, Diff: %lf\n", start/1000000., stop/1000000., (stop-start)/1000000.);
	  gettimeofday(&tv, NULL);
    stop = tv.tv_sec*1000000LL + tv.tv_usec;
    secs = (stop-start)/1000000.0;
    printf("Time taken: %lf\n", secs);
    printf("%lf GIOPS\n", (MAX_OPS*24.*threads)/(secs*1000000000.)); 
    
    
//    start = clock();
//    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i=0; i<threads; i++)
    {
       pthread_create(&cpu_threads[i], NULL, cpuFloat, NULL);
    }
    
    for(int i=0; i<threads; i++)
    {
        pthread_join(cpu_threads[i], NULL);
    }
//    stop = clock();
//    printf("Start Time: %lf, End Time: %lf, Diff: %lf\n", start/1000000., stop/1000000., (stop-start)/1000000.);
//    clock_gettime(CLOCK_MONOTONIC, &stop);
//	secs = timeDiff(&start, &stop);
//    printf("%lf GFLOPS\n", (MAX_OPS*24.*threads)/(((stop-start)/1000000.)*1000000000.)); 
	printf("Time taken: %lf\n", secs);
    printf("%lf GFLOPS\n", (MAX_OPS*24.*threads)/(secs*1000000000.)); 
    
    
    return 0;
}
