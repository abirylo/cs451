#!/bin/bash

FILE="/home/tli13/shared/synthetic_12GB"
LOGFILE="CPUCountTest.log"

touch $LOGFILE

for NUM_THREADS in 	{1..32}
do
	echo $NUM_THREADS Threads >> $LOGFILE
	{ time ./wordcountCPU $NUM_THREADS $FILE >> $LOGFILE; } 2>> $LOGFILE
	echo ================================================ >> $LOGFILE
	echo >> $LOGFILE
done

exit
