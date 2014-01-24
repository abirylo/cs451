CC = g++
CFLAGS = -c -Wall -std=c++11

all: cpu mem

cpu: benchCPU.o
	$(CC) benchCPU.o -o benchCPU -lpthread

benchCPU.o: benchCPU.cpp
	$(CC) $(CFLAGS) benchCPU.cpp

mem: benchMem.o
	$(CC) benchMem.o -o benchMem

benchMem.o: benchMem.cpp
	$(CC) $(CFLAGS) benchMem.cpp

clean:
	rm -rf *.o benchCPU benchMem
