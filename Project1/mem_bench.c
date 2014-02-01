#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

const size_t B = 1;
const size_t KB = 1024;
const size_t MB = 1048576;
const int NUM_LOOPS = 20;
const int CHUNK_SIZE = 1024; //size of memory to carve out, in MB
size_t size;
int MEM_SIZE;
char seqOrRand;

double timeDiff(struct timespec *start, struct timespec *end){
    return (double)(end->tv_sec-start->tv_sec)+((end->tv_nsec-start->tv_nsec)/1000000000.0);
}

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

void memTest(){
 	void* endptr;
    void* mem;
    int numOps = ((CHUNK_SIZE*MB)/size);

    mem = malloc(MB*CHUNK_SIZE);
	endptr = (char*)mem+(CHUNK_SIZE*MB);

	(seqOrRand == 'S') ? sequential(mem, endptr, size, numOps) : random_access(mem, endptr, size, numOps);
/*
    clock_gettime(CLOCK_MONOTONIC, &start);
	random_access(mem, endptr, size, numOps);
    clock_gettime(CLOCK_MONOTONIC, &stop);
    secs = timeDiff(&start, &stop);
    printf("Random access throughput=%f\n", (numOps*size)/(MB*secs));
    printf("Random access latency=%f\n", (secs*1000)/numOps);
*/
    free(mem);
        
    return;
}


int main(int argv, char* argc[]){
    //struct timespec start, stop;
    //double secs;

    if(argv != 4){
        fprintf(stderr, "Usage: benchMem <size (B, KB, MB)> <(S)equential or (R)andom> [number of threads]");
        return 1;
    }
    
    if(strcmp(argc[1],"B")==0){
        size = B;
    }else if(strcmp(argc[1],"KB")==0){
        size = KB;
    }else if(strcmp(argc[1],"MB")==0){
        size = MB;
    }

    seqOrRand = *argc[2];

    int threads = atoi(argc[3]);
    if(threads == 0){
        fprintf(stderr, "Specify the number of threads");
        return 1;
    }
    struct timeval tv;
    long long start, stop;
    double secs;
    
    pthread_t mem_threads[threads];
//    clock_gettime(CLOCK_MONOTONIC, &start);
    gettimeofday(&tv, NULL);
    start = tv.tv_sec*1000000LL + tv.tv_usec;
    for(int i=0; i<threads; i++){
        pthread_create(&mem_threads[i], NULL, (void *) &memTest, NULL);
    }
    for(int i=0; i<threads; i++){
        pthread_join(mem_threads[i], NULL);
    }
//    clock_gettime(CLOCK_MONOTONIC, &stop);
//    secs = timeDiff(&start, &stop);
    gettimeofday(&tv, NULL);
    stop = tv.tv_sec*1000000LL + tv.tv_usec;
    secs = (stop-start)/1000000.0;
    printf("Time taken: %lf\n", secs);
    printf("Throughput: %lf\n", (CHUNK_SIZE)/(secs));
    printf("Latency: %lf\n", (secs*1000)/((CHUNK_SIZE*MB)/size));

    return 0;
}
