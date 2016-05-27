#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-4 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)
let "n=${#instances_list[@]}-1"
cd /home/lisha/school/Hyperband/benchmarks
for k in $(seq 1 256); do 
	ssh -t ${instances_list[$k]}.us-central1-f.stately-will-107018 "sudo chmod -R 755 /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/"
done
wait
for k in $(seq 1 256); do 
	rsync -a -f"+ */" -f"+ *.pkl" -f"+ *.out" -f"- *" ${instances_list[$k]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_memory
done
wait
