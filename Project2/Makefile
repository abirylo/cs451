#Wordcount Makefile for CPU and GPU
CC=g++
FLAGS=-std=gnu++0x -pthread -Wl,--no-as-needed

all: CPU GPU

CPU: CPUwordcount.cpp
	$(CC) $(FLAGS) CPUwordcount.cpp -o wordcountCPU

GPU: GPUwordcount.cu book.h
	nvcc -arch=sm_20 -g -G GPUwordcount.cu -o wordcountGPU
