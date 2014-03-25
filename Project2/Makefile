#Wordcount Makefile for CPU and GPU
CC=g++
FLAGS=-std=gnu++0x -pthread -Wl,--no-as-needed

all: CPU GPU

CPU: wordcountCPU.cpp c_tokenizer.c
	$(CC) $(FLAGS) c_tokenizer.c wordcountCPU.cpp -o wordcountCPU

GPU: GPUwordcount.cu book.h
	nvcc -arch=sm_20 -g -G GPUwordcount.cu -o wordcountGPU
