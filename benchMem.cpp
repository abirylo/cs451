#include <stdio.h>
#include <time.h>
#include <string.h>

const size_t B = 1;
const size_t KB = 1024;
const size_t MB = 1048576;
const unsigned long NUMINTS = 2621440;

int main(int argv, char* argc[]){
    clock_t begin, end, clicks;
    float secs;
    size_t size = 0;
    int numLoops = 1;
    int* p;
    char* ptr1 = p;
    char* ptr2;

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
    for(int n=0; n<NUMINTS; n++){
        p = new int;
    }
    ptr2 = (ptr1+p)/2;

    begin = clock();
    for(int m=0; m<numLoops; m++){
        for(unsigned int i=0; i<ARRAY_SIZE*5; i+=size){
            memcpy(ptr1, ptr2, size);
            ptr1+=size;
            ptr2+=size;
        }
    }
    end = clock();
    clicks = end - begin;
    secs = ((float)clicks)/CLOCKS_PER_SEC;

    printf("Copied 2MB %d times using size %s, took %f secs.\n",numLoops, argc[1], secs);
    printf("Throughput: %f MB/Sec\n", (numLoops*ARRAY_SIZE)/(MB*secs));
    printf("Latency per operation: %4.2f ms\n", (secs*1000)/(ARRAY_SIZE));   
 
    return 0;
}
