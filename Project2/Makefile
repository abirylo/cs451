#Wordcount Makefile for CPU and GPU
CC=g++
FLAGS=-std=gnu++0x -pthread -Wl,--no-as-needed

all: CPU GPU

CPU: wordcountCPU.cpp c_tokenizer.c
	$(CC) $(FLAGS) c_tokenizer.c wordcountCPU.cpp -o wordcountCPU

GPU: wordcountGPU.cu hashtable.cu
	nvcc -arch=sm_20 -g -G hashtable.cu -dc
	nvcc -arch=sm_20 -g -G hashtable_d.cu -dc
	nvcc -arch=sm_20 -g -G wordcountGPU.cu -dc
	nvcc -arch=sm_20 -g -G -o wordcountGPU hashtable.o hashtable_d.o wordcountGPU.o

