#!/bin/bash

SMALLLOG="CPUCountTest2m.log"
SMALLFILE="2mdata.bin"
BIGLOG="CPUCountTest10g.log"
BIGFILE="10gdata.bin"

if [ "$1" == "big" ]
then
	LOGFILE=$BIGLOG
	BINFILE=$BIGFILE
elif [ "$1" == "small" ]
then
	LOGFILE=$SMALLLOG
	BINFILE=$SMALLFILE
else
	echo Wrong arguments.
	exit
fi

for NUM_THREADS in 192 384 768 1536 3072
do
	echo $NUM_THREADS Threads >> $LOGFILE
	{ time ./wordcountGPU $NUM_THREADS $BINFILE >> $LOGFILE; } 2>> $LOGFILE
	echo ================================================ >> $LOGFILE
	echo >> $LOGFILE
done

exit
