#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

const size_t B = 1;
const size_t KB = 1024;
const size_t MB = 1048576;
const int NUM_LOOPS = 10;
const int CHUNK_SIZE = 500; //size of memory to carve out, in MB

struct results_t {
    float throughput;
    float latency;
};

void sequential(void* lowptr, void* highptr, size_t size, int numOps){
	char *ptr1, *ptr2;
	ptr1 = (char*)lowptr;
	ptr2 = ptr1+size;
	
	for(int i=0; i<numOps; i++){
		memcpy(ptr1, ptr2, size);
		ptr1 += size;
		ptr2 += size;
	}

	return;
}

void random_access(void* lowptr, void* highptr, size_t size, int numOps){
	char *ptr1, *ptr2;
	int r1, r2;

	for(int i=0; i<numOps; i++){
		r1 = rand() % ((CHUNK_SIZE*MB)/size);
		r2 = rand() % ((CHUNK_SIZE*MB)/size);
		ptr1 = (char*)lowptr+r1;
		ptr2 = (char*)lowptr+r2;
		memcpy(ptr1, ptr2, size);
	}
	return;
}
results_t memTest(size_t size, char test){
    results_t results;
    clock_t begin, end, clicks;
    float secs;
 	void* endptr;
    void* mem;
    int numOps = (size == B) ? 52428800 : ((CHUNK_SIZE*MB)/size);

    mem = malloc(MB*CHUNK_SIZE);
	endptr = (char*)mem+(CHUNK_SIZE*MB);
//	printf("ptr1=%p\nptr2=%p\nendptr=%p\n",ptr1,ptr2,endptr);

    begin = clock();
	(test == 'S') ? sequential(mem, endptr, size, numOps) : random_access(mem, endptr, size, numOps);
    end = clock();
    clicks = end - begin;
    secs = ((float)clicks)/CLOCKS_PER_SEC;
    results.throughput = CHUNK_SIZE/secs;
    results.latency = numOps/(secs*1000);

    free(mem);
        
    return results;
}


int main(int argv, char* argc[]){
    size_t size = 0;
    results_t r;
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
//        printf("Throughput: %f MB/Sec\n", r.throughput);
//        printf("Latency per operation: %2.2f ms\n", r.latency);
    }

    avgThroughput /= NUM_LOOPS;
    avgLatency /= NUM_LOOPS;

    printf("The average throughput was: %f MB/sec\nThe average latency was: %f ms\n", avgThroughput, avgLatency);
	printf("The memory was accessed %s.\n", (SorR == 'S') ? "sequentially" : "randomly"); 

    return 0;
}
