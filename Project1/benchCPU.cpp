#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>

const unsigned long MAX_OPS = 2000000000;

void float_ops(unsigned long n){
    float a = 99.9999999999;
    float b = 999.999999999;
    float c = 9999.99999999;
    float d = 99999.9999999;
    unsigned long i;

    for(i=0; i<n; i++){
        a+=.99;
        b+=.99;
        c+=.99;
        d+=.99;
    }
    return;
}

void int_ops(unsigned long n){
    int a = -9999999;
    int b = 9999999;
    int c = 9999999;
    int d = 9999999;
    unsigned long i;

    for(i=0; i<n; i++){
        a+=1;
        b+=1;
        c+=1;
        d+=1;
    }
    return;
}

int main(int argc, char* argv[])
{
    int numThreads = 0;
    if(argc != 3){
        fprintf(stderr, "Usage: ./benchCPU -n [number of threads]\n");
        return 1;
    }
    if(strcmp(argv[1],"-n")==0){
        numThreads = atoi(argv[2]);
        if(numThreads == 0){
            fprintf(stderr, "Number of threads was not a valid number.\n");
            return 1;
        }
    }else{
        fprintf(stderr, "Usage: ./benchCPU -n [number of threads]\n");
        return 1;
    }

    unsigned long num_ops = MAX_OPS/numThreads;
    clock_t begin, end, time;
    double secs;
    std::vector<std::thread> threads;
    
    begin = clock();
    for(int i=0; i < numThreads; i++){
        threads.push_back(std::thread(float_ops, num_ops));
    }
  //  begin = clock();
    for(int i=0; i < numThreads; i++){
        (threads.back()).join();
        threads.pop_back();
    }
    end = clock();
    time = end - begin;
    secs = ((double)time)/CLOCKS_PER_SEC;

    printf("Began at %lu.  Ended at %lu.\nTook %lu clicks.  Took %f secs.\n",begin, end, time, secs);
    printf("%lf FLOPS.\n",(MAX_OPS*4*2)/(secs*1000000000)); //why *2???

    begin = clock();
    for(int i=0; i < numThreads; i++){
        threads.push_back(std::thread(int_ops, num_ops));
    }
    begin = clock();
    for(int i=0; i < numThreads; i++){
        (threads.back()).join();
        threads.pop_back();
    }
    end = clock();
    time = end - begin;
    secs = ((double)time)/CLOCKS_PER_SEC;

    printf("Began at %lu.  Ended at %lu.\nTook %lu clicks.  Took %f secs.\n",begin, end, time, secs);
    printf("%lf IOPS.\n",(MAX_OPS*4)/(secs*1000000000));


    return 0;
}
