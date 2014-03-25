#!/bin/bash

LOGFILE="GPUCountTest10g.log"
BINFILE="10gdata.bin"

touch $LOGFILE

for NUM_THREADS in 192 384 768 1536 3072
do
	echo $NUM_THREADS Threads >> $LOGFILE
	{ time ./wordcountGPU $NUM_THREADS $BINFILE >> $LOGFILE; } 2>> $LOGFILE
	echo ================================================ >> $LOGFILE
	echo >> $LOGFILE
done

exit
