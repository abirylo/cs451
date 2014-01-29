#~/bin/bash

LOG_FILE=bench.log

for a in {1..20}
do
	./benchCPU -n $a >> $LOG_FILE
done

exit
