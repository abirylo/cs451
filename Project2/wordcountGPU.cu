#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "hashtable.cuh"

#define CHECK_ERR(x)                                    \
  if (x != cudaSuccess) {                               \
    fprintf(stderr,"%s in %s at line %d\n",             \
      cudaGetErrorString(err),__FILE__,__LINE__); \
    exit(-1);           \
  }                                                     \

#define MEMORY_SIZE 268435456 //256Megabytes  
#define BLOCK_SIZE 128
  
__global__ void wordcount(char* words, char* result, long size, int n) {

  int a = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(a < n){
    //find start pointer offset
    char* start = (MEMORY_SIZE/BLOCK_SIZE) * (a) + words; 
    char* end = (MEMORY_SIZE/BLOCK_SIZE) * (a+1) + words;
    if(end > words+size)
      end = words+size;
    
    hashtable_t table;
    ht_init_s(&table, 1024);
    
// "?\";<>,~`!@#^&*()_+-=/\\:;{}[]|. "
    char *word_start = start;
    char *current_pos = start;
    while(current_pos <= end){
      if(*current_pos == '?' || *current_pos == '\"' || *current_pos == ';' || *current_pos == '<' || *current_pos == '>' || *current_pos == ',' || *current_pos == '~' || *current_pos == '`' || *current_pos == '!' || *current_pos == '@' || *current_pos == '#' || *current_pos == '^' || *current_pos == '&' || *current_pos == '*' || *current_pos == '(' || *current_pos == ')' || *current_pos == '_' || *current_pos == '+' || *current_pos == '-' || *current_pos == '=' || *current_pos == '/' || *current_pos == '\\' || *current_pos == ':' || *current_pos == '{' || *current_pos == '}' || *current_pos == '[' || *current_pos == ']' || *current_pos == '|' || *current_pos == '.' || *current_pos == ' ' || *current_pos == '\n' || *current_pos == '\0'){
        *current_pos = '\0';
        int val; 
        if((val = ht_get(&table, word_start)) != -1){
          ht_delete(&table, word_start);
          ht_add(&table, word_start, val+1);
        }
        else{
          ht_add(&table, word_start, 1);
        }
        word_start = current_pos+1;
      }
      current_pos++;
    }
    
    char *r = (MEMORY_SIZE/BLOCK_SIZE) * a + result;
    get_all_kvps(&table, (kvp_t*)r);
    //__syncthreads();
  
  }
}


int main(int argc, char *argv[]) {

  cudaError_t err;
  
  char* h_words;
  char* d_words;
  char* h_result;
  char* d_result;
  //char** h_dout;
  //char** d_dout;
  char* file_name = argv[1];
  FILE* file;
  long file_size = 0;
  long pos = 0;
  struct stat64 st;
  //char* contents;

  stat64(file_name, &st);
  file_size = st.st_size;

  file = fopen(file_name, "r");
  if(file == NULL)
    return -1;

  h_words = (char*)malloc(sizeof(char)*(MEMORY_SIZE>file_size ? file_size : MEMORY_SIZE)); 
  err = cudaMalloc((void **) &d_words, sizeof(char)*MEMORY_SIZE);
  CHECK_ERR(err);

  h_result = (char*)malloc(sizeof(char)*MEMORY_SIZE);
  err = cudaMalloc((void **) &d_result, sizeof(char)*MEMORY_SIZE);
  CHECK_ERR(err);
  
  while(pos < file_size){
    long read_size = read(file->_fileno, h_words, MEMORY_SIZE);
    pos += read_size;
    
    err = cudaMemcpy(d_words, h_words, read_size, cudaMemcpyHostToDevice);
    CHECK_ERR(err);
    
    wordcount<<< 1, BLOCK_SIZE >>>(d_words,d_result, read_size,(int)ceil((double)(read_size * BLOCK_SIZE)/(double)MEMORY_SIZE)); 
    
    err = cudaMemcpy(h_words, d_words, read_size, cudaMemcpyDeviceToHost);
    CHECK_ERR(err);
    err = cudaMemcpy(h_result, d_result, MEMORY_SIZE, cudaMemcpyDeviceToHost);
    CHECK_ERR(err);
    kvp_t *kvps = (kvp_t*)h_result; 
    char * c = (char *)(kvps[0].key-(long)d_words)+(long)h_words;
    printf("%s, %i", c, kvps[0].val);
  }



}
