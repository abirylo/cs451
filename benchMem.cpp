#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

const size_t B = 1;
const size_t KB = 1024;
const size_t MB = 1048576;
const int NUM_LOOPS = 20;

struct results_t {
    float throughput;
    float latency;
};

results_t memTest(size_t size){
    results_t results;
    clock_t begin, end, clicks;
    float secs;
    char *ptr1, *ptr2, *endptr;
    void* mem;
    int numOps = 0;

    mem = malloc(MB*100);
    ptr1 = (char*)mem;
    ptr2 = ptr1+size;
	endptr = (char*)((long)ptr1+100*MB);
//	printf("ptr1=%p\nptr2=%p\nendptr=%p\n",ptr1,ptr2,endptr);

    begin = clock();
        while(ptr1<endptr){
            memcpy(ptr1, ptr2, size);
//	printf("copying from %p to %p, operation number: %d\n",ptr1,ptr2,numOps);
            ptr1+=size;
            ptr2+=size;
		numOps++;
        }
    
    end = clock();
    clicks = end - begin;
    secs = ((float)clicks)/CLOCKS_PER_SEC;
    
    results.throughput = 100/secs;
    results.latency = numOps/(secs*1000);

    free(mem);
        
    return results;
}


int main(int argv, char* argc[]){
    size_t size = 0;
    results_t r;
    float avgThroughput=0.0;
    float avgLatency = 0.0;

    if(argv != 2){
        fprintf(stderr, "Please identify a block size (B, KB, or MB)");
        return 1;
    }
    
    if(strcmp(argc[1],"B")==0){
        size = B;
    }else if(strcmp(argc[1],"KB")==0){
        size = KB;
    }else if(strcmp(argc[1],"MB")==0){
        size = MB;
    }

    for(int i=0; i<NUM_LOOPS; i++){
        r = memTest(size);
        avgThroughput += r.throughput;
        avgLatency += r.latency;
//        printf("Throughput: %f MB/Sec\n", r.throughput);
//        printf("Latency per operation: %2.2f ms\n", r.latency);
    }

    avgThroughput /= NUM_LOOPS;
    avgLatency /= NUM_LOOPS;

    printf("The average throughput was: %f MB/sec\nThe average latency was: %f ms\n", avgThroughput, avgLatency);
 
    return 0;
}
