CC = gcc
CFLAGS = -c -Wall -std=gnu99 

all: cpu mem

cpu: cpu_bench.o
	$(CC) cpu_bench.o -o cpu_bench -lpthread

cpu_bench.o: cpu_bench.c
	$(CC) $(CFLAGS) cpu_bench.c

mem: mem_bench.o
	$(CC) mem_bench.o -o mem_bench

mem_bench.o: mem_bench.c
	$(CC) $(CFLAGS) mem_bench.c

clean:
	rm -rf *.o cpu_bench mem_bench
    

