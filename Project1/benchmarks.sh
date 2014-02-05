#!/bin/bash
LOGFILE=benchmarks.log

echo Benchmarking Suite Log >> $LOGFILE
date >> $LOGFILE

echo Running CPU Benchmarks...
echo Threads	GIOPS		GFLOPS >> $LOGFILE
for a in {1..16}
do
	./cpu_bench -n $a >> $LOGFILE
done
echo >> $LOGFILE

echo Running GPU Benchmanrks...
echo Type G*OPS >> $LOGFILE
for a in I F
do
  ./gpu_bench -t $a >> $LOGFILE
done
echo >> $LOGFILE
echo Size RW  Throughput >> $LOGFILE
for a in R W
do 
  for s in B K M
  do
    ./gpu_mem_bench -r $a -t $s >> $LOGFILE
  done
done
echo >> $LOGFILE

echo Running Memory Benchmarks...
echo Threads	Size	Rand/Seq	Throughput		Latency >> $LOGFILE
for a in {1..4}
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
for a in {1..1}
do
	for s in B KB MB GB
	do
		./disk_bench $s $a >> $LOGFILE
	done
done
echo >> $LOGFILE

echo Running Networking Benchmarsk...
echo Threads  Size  Protocol  Throughput  Latency >> $LOGFILE
for a in {1..2}
do 
  ./net_server_bench -n $a &
  sleep 2
  for s in B K 6
  do
    ./net_client_bench -n $a -l 20000 -t $s >> $LOGFILE
  done
  killall net_server_bench
  sleep 2
done
echo >> $LOGFILE

echo Done!

exit
