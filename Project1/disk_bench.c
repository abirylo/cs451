#include <stdio.h>
#include <time.h>
#include <string.h>

size_t size=0;
long numOps=0;

int main(int argc, char *argv[]){
    FILE *fp;
    char buff[100];
    struct timeval tv;
    long long start, stop;
    double secs;

    fp = fopen("temp", "rw+");
    if(fp == NULL){
        fp = fopen("temp", "a+");
        if(fp == NULL){
            fprintf(stderr, "File was not opened correctly.\n");
            return 1;
        }
    }
    if(argv[1] == "B"){
        size = 1;
        numOps = 1048576;
    }else if(argv[1] == "KB"){
        size = 1024;
        numOps = 102400;
    }else if(argv[1] == "MB"){
        size = 1048576;
        numOps = 1024
    }else if(argv[1] == "GB"){
        size = 1073741824;
        numOps = 4;
    }
    if(size == 0){
        fprintf(stderr, "Not a valid size.\n");
        return 1;
    }

    gettimeofday(&tv, NULL);
    start = tv.tv_sec*1000000LL + tv.tv_usec;
    write_seq(fp);
    gettimeofday(&tv, NULL);
    stop = tv.tv_sec*1000000LL + tv.tv_usec;
    secs = (stop-start)/1000000.0;

    printf("Time taken: %lf\n", secs);
    printf("Throughput: %lf MB/sec\n", (size*numOps)/(secs*1048576));
    printf("Latency: %lf ms\n", (secs*1000)/numOps);

    fclose(fp);
    return 0;
}
