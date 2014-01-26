CC = clang++
CFLAGS = -c -Wall
THREAD = -lpthread -std=c++11

all: cpu mem

cpu: benchCPU.o
	$(CC) benchCPU.o -o benchCPU $(THREAD)

benchCPU.o: benchCPU.cpp
	$(CC) $(CFLAGS) benchCPU.cpp $(THREAD)

mem: benchMem.o
	$(CC) benchMem.o -o benchMem $(THREAD)

benchMem.o: benchMem.cpp
	$(CC) $(CFLAGS) benchMem.cpp $(THREAD)

clean:
	rm -rf *.o benchCPU benchMem
