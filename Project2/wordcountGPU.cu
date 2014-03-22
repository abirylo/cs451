#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <string.h>
#include "hashtable_d.h"
#include "hashtable.h"

#define CHECK_ERR(x)                                    \
  if (x != cudaSuccess) {                               \
    fprintf(stderr,"%s in %s at line %d\n",             \
      cudaGetErrorString(err),__FILE__,__LINE__); \
    exit(-1);           \
  }                                                     \

#define MEMORY_SIZE 268435456 //256Megabytes  
#define BLOCK_SIZE 128
  
__global__ void wordcount(char* words, char* result, long* size_result, long size, int n) {

  int a = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(a < n){
    //find start pointer offset
    char* start = (MEMORY_SIZE/BLOCK_SIZE) * (a) + words; 
    char* end = (MEMORY_SIZE/BLOCK_SIZE) * (a+1) + words;
    if(end > words+size)
      end = words+size;
    
    hashtable_t_d table;
    ht_init_s_d(&table, (size_t)1024);
    
// "?\";<>,~`!@#^&*()_+-=/\\:;{}[]|. "
    char *word_start = start;
    char *current_pos = start;
    while(current_pos <= end){
      if(*current_pos == '?' || *current_pos == '\"' || *current_pos == ';' || *current_pos == '<' || *current_pos == '>' || *current_pos == ',' || *current_pos == '~' || *current_pos == '`' || *current_pos == '!' || *current_pos == '@' || *current_pos == '#' || *current_pos == '^' || *current_pos == '&' || *current_pos == '*' || *current_pos == '(' || *current_pos == ')' || *current_pos == '_' || *current_pos == '+' || *current_pos == '-' || *current_pos == '=' || *current_pos == '/' || *current_pos == '\\' || *current_pos == ':' || *current_pos == '{' || *current_pos == '}' || *current_pos == '[' || *current_pos == ']' || *current_pos == '|' || *current_pos == '.' || *current_pos == ' ' || *current_pos == '\n' || *current_pos == '\0'){
        *current_pos = '\0';
        int val; 
        if((val = ht_get_d(&table, word_start)) != -1){
          ht_delete_d(&table, word_start);
          ht_add_d(&table, word_start, val+1);
        }
        else{
          ht_add_d(&table, word_start, 1);
        }
        word_start = current_pos+1;
      }
      current_pos++;
    }
       
    char *r = (MEMORY_SIZE/BLOCK_SIZE) * a + result;
    get_all_kvps_d(&table, (kvp_t_d*)r);
    size_result[a] = table.size;
    
    ht_dispose_d(&table);
    //__syncthreads();
  
  }
}

int compareByVal(const void *a, const void *b){
  if(((kvp_t*)a)->val < ((kvp_t*)b)->val) return -1;
  if(((kvp_t*)a)->val == ((kvp_t*)b)->val) return 0;
  if(((kvp_t*)a)->val > ((kvp_t*)b)->val) return 1; 
}

int main(int argc, char *argv[]) {

  cudaError_t err;
  
  char* h_words;
  char* d_words;
  char* h_result;
  char* d_result;
  long* h_size_result;
  long* d_size_result;
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
  
  hashtable_t table;
  ht_init_s(&table, 1024*1024);
  
  h_words = (char*)malloc(sizeof(char)*(MEMORY_SIZE>file_size ? file_size : MEMORY_SIZE)); 
  err = cudaMalloc((void **) &d_words, sizeof(char)*MEMORY_SIZE);
  CHECK_ERR(err);

  h_result = (char*)malloc(sizeof(char)*MEMORY_SIZE);
  err = cudaMalloc((void **) &d_result, sizeof(char)*MEMORY_SIZE);
  CHECK_ERR(err);

  h_size_result = (long *)malloc(sizeof(long)*BLOCK_SIZE);
  err = cudaMalloc((void **) &d_size_result, sizeof(long)*BLOCK_SIZE);
  CHECK_ERR(err);

  while(pos < file_size){
    long read_size = read(file->_fileno, h_words, MEMORY_SIZE);
    pos += read_size;
    
    err = cudaMemcpy(d_words, h_words, read_size, cudaMemcpyHostToDevice);
    CHECK_ERR(err);
   
    int threads = (int)ceil((double)(read_size*BLOCK_SIZE)/(double)MEMORY_SIZE); 
    wordcount<<< 1, BLOCK_SIZE >>>(d_words,d_result,d_size_result,read_size,threads); 
    
    err = cudaMemcpy(h_words, d_words, read_size, cudaMemcpyDeviceToHost);
    CHECK_ERR(err);
    err = cudaMemcpy(h_result, d_result, threads*MEMORY_SIZE/BLOCK_SIZE, cudaMemcpyDeviceToHost);
    CHECK_ERR(err);
    err = cudaMemcpy(h_size_result, d_size_result, BLOCK_SIZE, cudaMemcpyDeviceToHost);
    CHECK_ERR(err);
    
    
    //*
    //need to merge
    for(int i=0; i<threads; i++){
      kvp_t *kvps = (kvp_t*)((MEMORY_SIZE/BLOCK_SIZE) * (i) + h_result); 
      for(int j=0; j<h_size_result[i]; j++){
        char * c = (char *)(kvps[j].key-(long)d_words)+(long)h_words;
        int val = 0;
        if((val = ht_get(&table, c)) != -1){
          ht_add(&table, (void *)c, val+kvps[j].val);
        }
        else{
          char* s = (char*)malloc(strlen(c));
          strcpy(s, c);
          ht_add(&table, s, kvps[j].val);
        }
      }
    }
    // */
  }

  err = cudaFree(d_words);
  CHECK_ERR(err);
  err = cudaFree(d_result);
  CHECK_ERR(err);
  err = cudaFree(d_size_result);
  CHECK_ERR(err);
  
  free(h_words);
  free(h_result);
  free(h_size_result);
  
  //*
  //print out the final hash table
  ht_delete(&table, "\0"); //delete null value 
  kvp_t* results = (kvp_t*)malloc(sizeof(kvp_t)*table.size);
  get_all_kvps(&table, (kvp_t*)results);
  qsort(results, table.size, sizeof(kvp_t), compareByVal);  
 
  for(int i=table.size-1; i>=0; i--){
    printf("%s %i\n",results[i].key, results[i].val);
    free((void *)results[i].key);
  }
  
  ht_dispose(&table);
  free(results);
  // */
}
