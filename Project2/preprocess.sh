#!/bin/bash

LOGFILE="processtime.log"
INPUT="/home/tli13/shared/int-dataset-10g.dat"
OUTPUT="10gdata.bin"
SMALL="/home/tli13/shared/int-dataset-2M.dat"
SMALLOUTPUT="2mdata.bin"

echo Processing $INPUT >> $LOGFILE
{ time cat $INPUT | python preprocess.py > $OUTPUT; } 2>> $LOGFILE

echo >> $LOGFILE
echo >> $LOGFILE

echo Processing $SMALL >> $LOGFILE
{ time cat $SMALL | python preprocess.py > $SMALLOUTPUT; } 2>> $LOGFILE

