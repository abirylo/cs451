CC = g++
CFLAGS = -c -Wall -std=c++11

all: cpu

cpu: benchCPU.o
	$(CC) benchCPU.o -o benchCPU -lpthread

benchCPU.o: benchCPU.cpp
	$(CC) $(CFLAGS) benchCPU.cpp

clean:
	rm -rf *.o benchCPU
