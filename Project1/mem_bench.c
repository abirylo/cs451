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
const int CHUNK_SIZE = 1048; //size of memory to carve out, in MB
size_t size;
int numOps;
int MEM_SIZE;
char seqOrRand;

double timeDiff(struct timespec *start, struct timespec *end){
    return (double)(end->tv_sec-start->tv_sec)+((end->tv_nsec-start->tv_nsec)/1000000000.0);
}

void sequential(void* lowptr, void* highptr, size_t size){
	char *ptr1, *ptr2;
	ptr1 = (char*)lowptr;
	ptr2 = ptr1+size;
	
        for(int i=0; i<numOps; i++){
		ptr1 = (char*)memcpy(ptr1, ptr2, size);
		ptr2 += size;
	}

	return;
}

void random_access(void* lowptr, void* highptr, size_t size){
	char *ptr1, *ptr2;
	int r;
    ptr1 = (char*)lowptr;

	for(int i=0; i<numOps; i++){
		r = rand() % (numOps);
		ptr2 = (char*)lowptr+((r==0) ? r+1 : r)*size;
		memcpy(ptr1, ptr2, size);
	}
	return;
}

void memTest(){
 	void* endptr;
    void* mem;

    mem = malloc(MB*CHUNK_SIZE);
	endptr = (char*)mem+(CHUNK_SIZE*MB);

	(seqOrRand == 'S') ? sequential(mem, endptr, size) : random_access(mem, endptr, size);
    free(mem);
        
    return;
}


int main(int argv, char* argc[]){

    if(argv != 4){
        fprintf(stderr, "Usage: benchMem <size (B, KB, MB)> <(S)equential or (R)andom> [number of threads]");
        return 1;
    }
    
    if(strcmp(argc[1],"B")==0){
        size = B;
	numOps = 10485760;
    }else if(strcmp(argc[1],"KB")==0){
        size = KB;
	numOps = 1048756;
    }else if(strcmp(argc[1],"MB")==0){
        size = MB;
	numOps = 1024;
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
   printf("%d\t%s\t%c\t", threads, argc[1], seqOrRand); 
    pthread_t mem_threads[threads];
    gettimeofday(&tv, NULL);
    start = tv.tv_sec*1000000LL + tv.tv_usec;
    for(int i=0; i<threads; i++){
        pthread_create(&mem_threads[i], NULL, (void *) &memTest, NULL);
    }
    for(int i=0; i<threads; i++){
        pthread_join(mem_threads[i], NULL);
    }
    gettimeofday(&tv, NULL);
    stop = tv.tv_sec*1000000LL + tv.tv_usec;
    secs = (stop-start)/1000000.0;
//    printf("Time taken: %lf\n", secs);
//    printf("Throughput: %lf\n", (CHUNK_SIZE*threads)/(secs));
//    printf("Latency: %lf\n", (secs*1000)/(CHUNK_SIZE*MB*threads*8));
	printf("%lf\t%lf\n", (size*numOps*threads)/(secs*MB), (secs*1000)/(numOps*threads));
    return 0;
}
