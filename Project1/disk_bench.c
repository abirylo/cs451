#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

size_t size=0;
long numOps=0;

void write_seq(FILE *fd){
	void *buff;
	buff = malloc(size);
	memset(buff, 'S', size);
	setvbuf(fd, NULL, _IONBF, size);

	for(int i=0; i<numOps; i++){
		fwrite(buff, size, 1, fd);
//		fflush();
	}

	free(buff);

	return;	
}

void write_rand(FILE *fd){
	void *buff;
	long r;
	buff = malloc(size);
	memset(buff, 'R', size);
	setvbuf(fd, NULL, _IONBF, size);

	for(int i=0; i<numOps; i++){
		r = rand() % numOps;
		fseek(fd, r*size, SEEK_SET);
		fwrite(buff, size, 1, fd);
	}

	free(buff);

	return;
}

void read_seq(FILE *fd){
	void *buff;
	buff = malloc(size);
	setvbuf(fd, NULL, _IONBF, size);
	
	for(int i=0; i<numOps; i++){
//		buff = malloc(size);
		fread(buff, size, 1, fd);
//		free(buff);
	}

	free(buff);

	return;
}

void read_rand(FILE *fd){
	void* buff;
	long r;
	buff = malloc(size);
	setvbuf(fd, NULL, _IONBF, size);

	for(int i=0; i<numOps; i++){
		r = rand() % numOps;
		fseek(fd, r*size, SEEK_SET);
		fread(buff, size, 1, fd);
	}

	free(buff);

	return;
}

void clear_cache(){
	int fd;
	char* data = "3";

	sync();
	fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
	write(fd, data, sizeof(char));
	close(fd);
}

int main(int argc, char *argv[]){
	int threads=1; 
    struct timeval tv;
    long long start, stop;
    double secs;

    if(strcmp(argv[1],"B")==0){
        size = 1;
        numOps = 10485760;
    }else if(strcmp(argv[1], "KB")==0){
        size = 1024;
        numOps = 1048576;
    }else if(strcmp(argv[1], "MB")==0){
        size = 1048576;
        numOps = 1024;
    }else if(strcmp(argv[1], "GB")==0){
        size = 1073741824;
        numOps = 4;
    }
    if(size == 0){
        fprintf(stderr, "Not a valid size.\n");
        return 1;
    }

	threads = atoi(argv[2]);
	pthread_t disk_threads[threads];
	FILE *fd[threads];
	char buff[8];
	for(int i=0; i<threads; i++){
		sprintf(buff, "temp%d", i);
    	fd[i] = fopen(buff, "w+");
	}

	printf("%d\t%s\t", threads, argv[1]);

    gettimeofday(&tv, NULL);
    start = tv.tv_sec*1000000LL + tv.tv_usec;
	for(int i=0; i<threads; i++){
    		pthread_create(&disk_threads[i], NULL, (void *)&write_seq, fd[i]);
	}
	for(int i=0; i<threads; i++){
		pthread_join(disk_threads[i], NULL);
	}
    gettimeofday(&tv, NULL);
    stop = tv.tv_sec*1000000LL + tv.tv_usec;
    secs = (stop-start)/1000000.0;
	
//	printf("Sequential Write:\n");
//    printf("Time taken: %lf\n", secs);
	printf("%lf\t%lf\t", (size*numOps*threads)/(secs*1048576), (secs*1000)/(numOps*threads*size*8));
//    printf("Throughput: %lf MB/sec\n", (size*numOps*threads)/(secs*1048576));
//    printf("Latency: %lf ms/bit\n", (secs*1000)/(numOps*threads*size*8));

	for(int i=0; i<threads; i++){
		fclose(fd[i]);
	}
	clear_cache();

	for(int i=0; i<threads; i++){
		sprintf(buff, "temp%d", i);
    		fd[i] = fopen(buff, "r+");
	}

    gettimeofday(&tv, NULL);
    start = tv.tv_sec*1000000LL + tv.tv_usec;
	for(int i=0; i<threads; i++){
    		pthread_create(&disk_threads[i], NULL, (void *)&read_seq, fd[i]);
	}
	for(int i=0; i<threads; i++){
		pthread_join(disk_threads[i], NULL);
	}
    gettimeofday(&tv, NULL);
    stop = tv.tv_sec*1000000LL + tv.tv_usec;
    secs = (stop-start)/1000000.0;
	
//	printf("Sequential Read:\n");
//    printf("Time taken: %lf\n", secs);
//    printf("Throughput: %lf MB/sec\n", (size*numOps*threads)/(secs*1048576));
//    printf("Latency: %lf ms/bit\n", (secs*1000)/(numOps*threads*size*8));
	printf("%lf\t%lf\t", (size*numOps*threads)/(secs*1048576), (secs*1000)/(numOps*threads*size*8));

	for(int i=0; i<threads; i++){
		fclose(fd[i]);
	}
	clear_cache();
	numOps = (size > 1024) ? numOps : 102400;
	for(int i=0; i<threads; i++){
		sprintf(buff, "temp%d", i);
    	fd[i] = fopen(buff, "w+");
	}

    gettimeofday(&tv, NULL);
    start = tv.tv_sec*1000000LL + tv.tv_usec;
	for(int i=0; i<threads; i++){
    		pthread_create(&disk_threads[i], NULL, (void *)&write_rand, fd[i]);
	}
	for(int i=0; i<threads; i++){
		pthread_join(disk_threads[i], NULL);
	}
    gettimeofday(&tv, NULL);
    stop = tv.tv_sec*1000000LL + tv.tv_usec;
    secs = (stop-start)/1000000.0;
	
//	printf("Random Write:\n");
//    printf("Time taken: %lf\n", secs);
//    printf("Throughput: %lf MB/sec\n", (size*numOps*threads)/(secs*1048576));
//    printf("Latency: %lf ms/bit\n", (secs*1000)/(numOps*threads*size*8));
	printf("%lf\t%lf\t", (size*numOps*threads)/(secs*1048576), (secs*1000)/(numOps*threads*size*8));

	for(int i=0; i<threads; i++){
		fclose(fd[i]);
	}
	clear_cache();
	for(int i=0; i<threads; i++){
		sprintf(buff, "temp%d", i);
    	fd[i] = fopen(buff, "r+");
	}

    gettimeofday(&tv, NULL);
    start = tv.tv_sec*1000000LL + tv.tv_usec;
	for(int i=0; i<threads; i++){
    		pthread_create(&disk_threads[i], NULL, (void *)&read_rand, fd[i]);
	}
	for(int i=0; i<threads; i++){
		pthread_join(disk_threads[i], NULL);
	}
    gettimeofday(&tv, NULL);
    stop = tv.tv_sec*1000000LL + tv.tv_usec;
    secs = (stop-start)/1000000.0;
	
//	printf("Random Read:\n");
//    printf("Time taken: %lf\n", secs);
//    printf("Throughput: %lf MB/sec\n", (size*numOps*threads)/(secs*1048576));
//    printf("Latency: %lf ms/bit\n", (secs*1000)/(numOps*threads*size*8));
	printf("%lf\t%lf\n", (size*numOps*threads)/(secs*1048576), (secs*1000)/(numOps*threads*size*8));

	for(int i=0; i<threads; i++){
		fclose(fd[i]);
	}
	for(int i=0; i<threads; i++){
		sprintf(buff, "temp%d", i);
    		remove(buff);
	}
    return 0;
}
