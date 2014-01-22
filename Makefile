CC = g++
CFLAGS = -c -Wall

all: cpu

cpu: benchCPU.o
	$(CC) benchCPU.o -o benchCPU

benchCPU.o: benchCPU.cpp
	$(CC) $(CFLAGS) benchCPU.cpp

clean:
	rm -rf *.o benchCPU
