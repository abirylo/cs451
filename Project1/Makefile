CC = gcc
CFLAGS = -c -Wall -std=gnu99 -g 

all: net_client net_server cpu mem gpu gpu_mem disk

cpu: cpu_bench.o
	$(CC) cpu_bench.o -o cpu_bench -lpthread

cpu_bench.o: cpu_bench.c
	$(CC) $(CFLAGS) cpu_bench.c

mem: mem_bench.o
	$(CC) mem_bench.o -o mem_bench -lpthread

mem_bench.o: mem_bench.c
	$(CC) $(CFLAGS) mem_bench.c

disk: disk_bench.o
	$(CC) disk_bench.o -o disk_bench -lpthread

disk_bench.o: disk_bench.c
	$(CC) $(CFLAGS) disk_bench.c

gpu:
	nvcc -o gpu_bench gpu_bench.cu 

gpu_mem:
	nvcc -o gpu_mem_bench gpu_mem_bench.cu

net_server: net_server.o
	$(CC) net_server_bench.o -o net_server_bench -lpthread

net_server.o: net_server_bench.c
	$(CC) $(CFLAGS) net_server_bench.c

net_client: net_client.o
	$(CC) net_client_bench.o -o net_client_bench -lpthread

net_client.o: net_client_bench.c
	$(CC) $(CFLAGS) net_client_bench.c

clean:
	rm -rf *.o cpu_bench mem_bench gpu_bench gpu_mem_bench disk_bench net_server_bench net_client_bench
    

