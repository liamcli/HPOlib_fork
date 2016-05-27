#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-2 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)
let "n=${#instances_list[@]}-1"
cd /home/lisha/school/Hyperband/benchmarks
for k in $(seq 1 2); do 
	ssh -t ${instances_list[$k]}.us-central1-f.stately-will-107018 "sudo find /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/* -name \"*.out\" | xargs -0 chmod 666"
done
wait
for k in $(seq 1 2); do 
	rsync -a -f"+ */" -f"+ open_ml*.out" -f"- *" --exclude -f"+ */" -f"+ instance*.out" -f"- *" ${instances_list[$k]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_seeds2
done
wait
