#Wordcount Makefile for CPU and GPU
CC=g++
FLAGS=-std=gnu++11 -pthread -Wl,--no-as-needed

all: CPU

CPU: wordcountCPU.cpp c_tokenizer.c
	$(CC) $(FLAGS) c_tokenizer.c wordcountCPU.cpp -o wordcountCPU
