#!/bin/bash

FILE="/home/tli13/shared/synthetic_12GB"
LOGFILE="CPUCountTest.log"
BINFILE="2mdata.bin"

touch $LOGFILE

for NUM_THREADS in 1 2 4 8 16 32
do
	echo $NUM_THREADS Threads >> $LOGFILE
	{ time ./wordcountCPU $NUM_THREADS $BINFILE >> $LOGFILE; } 2>> $LOGFILE
	echo ================================================ >> $LOGFILE
	echo >> $LOGFILE
done

exit
