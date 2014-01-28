#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>

const int M = 500;
const int N = 500;
const int P = 500;

void int_mult(int A[M][N], int B[N][P]){
	int C[M][P];
	for(int i = 0; i<M; i++){
		for(int j = 0; j<N; j++){
			int sum = 0;
			for(int k = 0; k<P; k++){
				sum += A[i][k] * B[k][j];
			}
		C[i][j] = sum;
		}
	}
	return;
}

void float_mult(float A[M][N], float B[N][P]){
	float C[M][P];
	for(int i = 0; i<M; i++){
		for(int j = 0; j<N; j++){
			float sum = 0;
			for(int k = 0; k<P; k++){
				sum += A[i][k] * B[k][j];
			}
		C[i][j] = sum;
		}
	}
	return;
}

void float_ops(int numThreads){
	struct timespec start, finish;
	float A[M][N], B[M][N];
	std::vector<std::thread> threads;
	float numOps = (float)(M*N*P);
	numOps *= numThreads;

	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			A[i][j] = (float)rand();
			B[i][j] = (float)rand();
		}
	}
	
	clock_gettime(CLOCK_MONOTONIC, &start);
	for(int i=0; i<numThreads; i++){
		threads.push_back(std::thread(float_mult, A, B));
	}
	while(!threads.empty()){
		(threads.back()).join();
		threads.pop_back();
	}
	clock_gettime(CLOCK_MONOTONIC, &finish);
	double secs = (finish.tv_sec - start.tv_sec);
	secs += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	float FLOPS = (numOps)/(secs);
	FLOPS /= 1000000000;

	printf("%f\t", FLOPS);

    return;
}

void int_ops(int numThreads){
	int A[M][N], B[M][N];
	struct timespec start, finish;
	std::vector<std::thread> threads;
	float numOps = (float)(M*N*P);
	numOps *= numThreads;

	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			A[i][j] = (int)rand();
			B[i][j] = (int)rand();
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &start);
	for(int i=0; i<numThreads; i++){
		threads.push_back(std::thread(int_mult, A, B));
	}
	while(!threads.empty()){
		(threads.back()).join();
		threads.pop_back();
	}
	clock_gettime(CLOCK_MONOTONIC, &finish);
	double secs = (finish.tv_sec - start.tv_sec);
	secs += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	float IOPS = (numOps)/(secs);
	IOPS /= 1000000000;
	printf("%f\n", IOPS);

    return;
}

int main(int argc, char* argv[])
{
    int numThreads = 0;
    if(argc != 3){
        fprintf(stderr, "Usage: ./benchCPU -n [number of threads]\n");
        return 1;
    }
    if(strcmp(argv[1],"-n")==0){
        numThreads = atoi(argv[2]);
        if(numThreads == 0){
            fprintf(stderr, "Number of threads was not a valid number.\n");
            return 1;
        }
    }else{
        fprintf(stderr, "Usage: ./benchCPU -n [number of threads]\n");
        return 1;
    }
	printf("%d\t",numThreads);
	float_ops(numThreads);
	int_ops(numThreads);

    return 0;
}
