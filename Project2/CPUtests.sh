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

for NUM_THREADS in 1 2 4 8 16 32
do
	echo $NUM_THREADS Threads >> $LOGFILE
	{ time ./wordcountCPU $NUM_THREADS $BINFILE >> $LOGFILE; } 2>> $LOGFILE
	echo ================================================ >> $LOGFILE
	echo >> $LOGFILE
done

exit
