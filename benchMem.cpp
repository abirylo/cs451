#include <stdio.h>
#include <time.h>
#include <string.h>

const size_t B = 1;
const size_t KB = 1024;
const size_t MB = 1048576;
const unsigned long ARRAY_SIZE = 2097152; //number of bytes in 2MB

int main(int argv, char* argc[]){
    clock_t begin, end, clicks;
    float secs;
    int numLoops = 1000;

    char array1[ARRAY_SIZE];
    char array2[ARRAY_SIZE];

    begin = clock();
    for(int m=0; m<numLoops; m++){
        char* ptr1 = &array1[0];
        char* ptr2 = &array2[0];
        for(unsigned int i=0; i<ARRAY_SIZE; i+=B){
            memcpy(ptr1, ptr2, B);
            ptr1+=B;
            ptr2+=B;
        }
    }
    end = clock();
    clicks = end - begin;
    secs = ((float)clicks)/CLOCKS_PER_SEC;

    printf("Copied 2MB %d times using size B, took %f secs.\n",numLoops, secs);
    printf("Throughput: %f MB/Sec\n", (numLoops*ARRAY_SIZE)/(MB*secs));
    printf("Latency per operation: %4.2f ms\n", (secs*1000)/(numLoops*ARRAY_SIZE));   
 
    return 0;
}
