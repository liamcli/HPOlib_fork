#!/bin/bash
#output=$(gcloud compute instance-groups managed list-instances instance-group-1 --zone us-central1-f | cut -d' ' -f1)
#output=$(gcloud compute instances list --zone us-central1-f | cut -d' ' -f1)
#instances_list=($output)
tids=$(cut -d',' -f1 to_fix.csv)
tids=($tids)
searchers=$(cut -d',' -f2 to_fix.csv)
searchers=($searchers)
let "n=${#tids[@]}-1"
echo $n
cd /home/lisha/school/Hyperband/benchmarks/autosklearn
#seeds="1 5001 10001 15001 20001 25001 30001 35001 40001 45001"
for k in $(seq 0 $n); do 
	s=$(echo ${searchers[$k]} | cut -d'/' -f2)
	echo $s
	echo ${tids[$k]}
	rm -rf open_ml_${tids[$k]}_"$s"_*

done
wait
