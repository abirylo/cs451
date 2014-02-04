#!/bin/bash
LOGFILE=benchmarks.log

echo Benchmarking Suite Log >> $LOGFILE
date >> $LOGFILE

echo Running CPU Benchmarks...
echo Threads	GIOPS		GFLOPS >> $LOGFILE
for a in {1..20}
do
	./cpu_bench -n $a >> $LOGFILE
done
echo >> $LOGFILE

echo Running Memory Benchmarks...
echo Threads	Size	Rand/Seq	Throughput		Latency >> $LOGFILE
for a in {1..20}
do
	for s in B KB MB
	do
		./mem_bench $s S $a >> $LOGFILE
		./mem_bench $s R $a >> $LOGFILE
	done
done
echo >> $LOGFILE

echo Running Disk Benchmarks...
echo Threads	Size	Seq Write T	Seq Write L	Seq Read T	Seq Read L	Rand Write T	Rand Write L	Rand Read T	Rand Read L >> $LOGFILE
for a in {1..20}
do
	for s in B KB MB GB
	do
		./disk_bench $s $a >> $LOGFILE
	done
done
echo >> $LOGFILE

echo Done!

exit
