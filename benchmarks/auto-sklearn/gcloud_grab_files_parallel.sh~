#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-2 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)
let "n=${#instances_list[@]}-1"
cd /home/lisha/school/Hyperband/benchmarks

let "k=0"
for tid in $(seq 0 58); do 
	let "k1=$k+1"
	let "k2=$k+2"
	let "k3=$k+3"
	let "k4=$k+4"
	let "k5=$k+5"
	let "k6=$k+6"
	let "k7=$k+7"
	let "k8=$k+8"		
	ssh -t ${instances_list[$k1]}.us-central1-f.stately-will-107018 "sudo chmod -R 755 /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/" &
	ssh -t ${instances_list[$k2]}.us-central1-f.stately-will-107018 "sudo chmod -R 755 /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/" &
	ssh -t ${instances_list[$k3]}.us-central1-f.stately-will-107018 "sudo chmod -R 755 /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/" &
	ssh -t ${instances_list[$k4]}.us-central1-f.stately-will-107018 "sudo chmod -R 755 /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/" &
	ssh -t ${instances_list[$k5]}.us-central1-f.stately-will-107018 "sudo chmod -R 755 /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/" &
	ssh -t ${instances_list[$k6]}.us-central1-f.stately-will-107018 "sudo chmod -R 755 /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/" &
	ssh -t ${instances_list[$k7]}.us-central1-f.stately-will-107018 "sudo chmod -R 755 /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/" &
	ssh -t ${instances_list[$k8]}.us-central1-f.stately-will-107018 "sudo chmod -R 755 /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/" 
	let "k=$k1+7"
	wait
done
wait

let "k=0"
for tid in $(seq 0 58); do 
	let "k1=$k+1"
	let "k2=$k+2"
	let "k3=$k+3"
	let "k4=$k+4"
	let "k5=$k+5"
	let "k6=$k+6"
	let "k7=$k+7"
	let "k8=$k+8"
	rsync -a -f"+ */" -f"+ *.pkl" -f"+ *.out" -f"- *" ${instances_list[$k1]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_seeds3 & 
	rsync -a -f"+ */" -f"+ *.pkl" -f"+ *.out" -f"- *" ${instances_list[$k2]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_seeds3 &
	rsync -a -f"+ */" -f"+ *.pkl" -f"+ *.out" -f"- *" ${instances_list[$k3]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_seeds3 &
	rsync -a -f"+ */" -f"+ *.pkl" -f"+ *.out" -f"- *" ${instances_list[$k4]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_seeds3 &
	rsync -a -f"+ */" -f"+ *.pkl" -f"+ *.out" -f"- *" ${instances_list[$k5]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_seeds3 &
	rsync -a -f"+ */" -f"+ *.pkl" -f"+ *.out" -f"- *" ${instances_list[$k6]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_seeds3 &
	rsync -a -f"+ */" -f"+ *.pkl" -f"+ *.out" -f"- *" ${instances_list[$k7]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_seeds3 &
	rsync -a -f"+ */" -f"+ *.pkl" -f"+ *.out" -f"- *" ${instances_list[$k8]}.us-central1-f.stately-will-107018:/home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv/ autosklearn/holdout_seeds3 
	let "k=$k1+7"
	wait
done
wait
