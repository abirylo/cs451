#include <stdio.h>
#include <time.h>
#include <string.h>

const size_t B = 1;
const size_t KB = 1024;
const size_t MB = 1048576;
const unsigned long NUMINTS = 2621440; //number of ints in 10MB

int main(int argv, char* argc[]){
    clock_t begin, end, clicks;
    float secs;
    size_t size = 0;
    int* p = new int;
    char* ptr1 = (char*)p-4;
    char* ptr2;
	char* endptr;
	int numOps = 0;

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
    for(int n=0; n<NUMINTS*20; n++){
        p = new int;
    }
    ptr2 = ptr1+size;
	endptr = (char*)((long)ptr1+200*MB);
//	printf("ptr1=%p\nptr2=%p\np=%p\nendptr=%p\n",ptr1,ptr2,p,endptr);

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

    printf("Copied 200MB using size %s, took %f secs.\nnumOps = %d\n", argc[1], secs, numOps);
    printf("Throughput: %f MB/Sec\n", (10/secs));
    printf("Latency per operation: %2.2f ms\n", numOps/(secs*1000));   
 
    return 0;
}
