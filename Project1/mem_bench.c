#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

const size_t B = 1;
const size_t KB = 1024;
const size_t MB = 1048576;
const int NUM_LOOPS = 20;
const int CHUNK_SIZE = 1024; //size of memory to carve out, in MB

struct results_t {
    float throughput;
    float latency;
};

void sequential(void* lowptr, void* highptr, size_t size, int numOps){
	char *ptr1, *ptr2;
	ptr1 = (char*)lowptr;
	ptr2 = ptr1+size;
	
	for(int i=0; i<numOps; i++){
        ptr1 = (char*)memcpy(ptr1, ptr2, size);
		ptr2 += size;
	}

	return;
}

void random_access(void* lowptr, void* highptr, size_t size, int numOps){
	char *ptr1, *ptr2;
	int r;
    ptr1 = (char*)lowptr;

	for(int i=0; i<numOps; i++){
		r = rand() % ((CHUNK_SIZE*MB)/size);
		ptr2 = (char*)lowptr+((r==0) ? r+1 : r)*size;
		memcpy(ptr1, ptr2, size);
	}
	return;
}

struct results_t memTest(size_t size, char test){
    struct results_t results;
 //   &results = malloc(sizeof(struct results_t));
    clock_t begin, end, clicks;
    float secs;
 	void* endptr;
    void* mem;
    int numOps = (size == B) ? 52428800 : ((CHUNK_SIZE*MB)/size);

    mem = malloc(MB*CHUNK_SIZE);
	endptr = (char*)mem+(CHUNK_SIZE*MB);

    begin = clock();
	(test == 'S') ? sequential(mem, endptr, size, numOps) : random_access(mem, endptr, size, numOps);
    end = clock();
    clicks = end - begin;
    secs = ((float)clicks)/CLOCKS_PER_SEC;
//    printf("bytes moved = %lu\n", (numOps*size));
    results.throughput = (numOps*size)/(MB*secs);
    results.latency = (secs*1000)/numOps;

    free(mem);
        
    return results;
}


int main(int argv, char* argc[]){
    size_t size = 0;
    struct results_t r;
    float avgThroughput=0.0;
    float avgLatency = 0.0;

    if(argv != 3){
        fprintf(stderr, "Usage: benchMem <size (B, KB, MB)> <(S)equential, (R)andom>");
        return 1;
    }
    
    if(strcmp(argc[1],"B")==0){
        size = B;
    }else if(strcmp(argc[1],"KB")==0){
        size = KB;
    }else if(strcmp(argc[1],"MB")==0){
        size = MB;
    }

	char SorR = *argc[2];

    for(int i=0; i<NUM_LOOPS; i++){
        r = memTest(size, SorR);
        avgThroughput += r.throughput;
        avgLatency += r.latency;
    }

    avgThroughput /= NUM_LOOPS;
    avgLatency /= NUM_LOOPS;

    printf("The average throughput was: %f MB/sec\nThe average latency was: %f ms\n", avgThroughput, avgLatency);
	printf("The memory was accessed %s.\n", (SorR == 'S') ? "sequentially" : "randomly"); 

    return 0;
}
