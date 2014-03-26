#include <stdio.h>
#include <mutex>
#include <thread>
#include <algorithm>
#include <vector>

#define ELEMENTS 65535
#define FILE_BUFF 1024*1024*100  //10mb file buffer

//global pointers, for all threads
unsigned int table[ELEMENTS];
std::mutex locks[ELEMENTS];
unsigned short* keys;
int num_keys;
int numThreads;
int keys_per_thread;

typedef std::pair<int, int> TKeyValPair;

void put(int key){
	locks[key].lock();
	table[key]++;
	locks[key].unlock();
}

void t_process(int tid){
//	printf("Thread %d reporting in!\n", tid);
	int start_key = tid*keys_per_thread;
//	printf("Starting at element %d\n", start_key);
	for(int i=0; i<keys_per_thread; i++){
		unsigned short key = keys[start_key+i];
		if(key != 0) put(key);
	}
}

void test_buffer(unsigned short* buffer, size_t buffer_size){
	unsigned short value;
	for(int i=0; i<buffer_size/2; i++){
		value = buffer[i];
		printf("value = %u\n", value);
	}
}

void output_table(){
	FILE *outFile;
	outFile = fopen("CPUoutput.log", "w");
	for(int i=0; i<ELEMENTS; i++){
		fprintf(outFile, "key = %d ",i);
		fprintf(outFile, "value = %u\n",table[i]);
	}
	fclose(outFile);
}

bool comp(TKeyValPair i, TKeyValPair j){
	return ((i.second)>(j.second));
}

void sorted_output(){
	std::vector<TKeyValPair> values;
	TKeyValPair temp;
	long totalWords = 0;
	for(int i=0; i<ELEMENTS; i++){
		temp.first = i;
		temp.second = table[i];
		totalWords += temp.second;
		values.push_back(temp);
	}
	std::sort(values.begin(), values.end(), comp);
	FILE *ofd;
	ofd = fopen("Top50.out", "w");
	fprintf(ofd, "Total Words: %ld\n\n", totalWords);
	for(int i=0; i<50; i++){
		temp = values.at(i);
		fprintf(ofd, "%d\t%d\n", temp.first, temp.second);
	}
	fclose(ofd);
}

int main(int argc, char* argv[]){
	numThreads = atoi(argv[1]);
	size_t size_read;
	FILE *fd;
	std::vector<std::thread> threads;

	fd = fopen(argv[2], "rb");
	if(fd == NULL){
		fputs("File Error.", stderr);
		exit(1);
	}

	keys = (unsigned short*)malloc(FILE_BUFF);
	size_read = fread(keys, 1, FILE_BUFF, fd);
	num_keys = size_read/2;
	keys_per_thread = num_keys/numThreads;

	while(size_read !=0){
		for(int i=0; i<numThreads; i++){
			threads.push_back(std::thread(t_process, i));
		}

		for(std::vector<std::thread>::iterator it = threads.begin(); it != threads.end(); it++){
			it->join();
		}
		threads.clear();
		size_read = fread(keys, 1, FILE_BUFF, fd);
		num_keys = size_read/2;
		keys_per_thread = num_keys/numThreads;
	}

	output_table();

//	sorted_output(); Uncomment to get top 50 occurances

	free(keys);
	fclose(fd);
}

