#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-2 --zone us-east1-b | cut -d' ' -f1)
instances_list=($output)
tids=(3863  3865  3881  3882  3883  3884  3889  3893  3894  3902  3903  3904  3907  3917  3918  3919  3945  3946  3948  3950  3953  3954  3962  3964  3968  3972  3973  3976  3980  3995  4000  12  14  16  2071  2072  2073  2074  2076  2077  18  21  22  23  24  26  2270  28  3020  3481  30  31  32  3505  3506  3507  36  3524  3021  3525  3526  3527  3528  3530  3531  3533  3534  3  3536  43)
seeds="1 5001 10001 15001 20001 25001 30001 35001 40001 45001"
seeds2="50001 55001 60001 65001 70001 75001 80001 85001 90001 95001"
seeds3="100001 105001 110001 115001 120001 125001 130001 135001 140001 145001"
dataseeds="1 1 1 1 1 1 1 1 1 1"
for k in $(seq 1 210); do 
	let "optimizer=($k-1)%3"
	let "tid = ($k-1)/3"
	echo $optimizer
	echo $tid
	#if [ $optimizer -eq 0 ]; then
	#	gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-east1-b --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/hyperopt ${tids[$tid]} \"$seeds\" \"$dataseeds\"" 
	#fi
	#if [ $optimizer -eq 1 ]; then
	#	gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-east1-b --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh smac/smac ${tids[$tid]} \"$seeds\" \"$dataseeds\"" 
	#fi
	if [ $optimizer -eq 2 ]; then
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-east1-b --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh nvb_hyperband/nvb_hyperband ${tids[$tid]} \"$seeds\" \"$dataseeds\"" 
	fi

done
wait

#for k in $(seq 211 350); do 
#	let "optimizer=($k-211)%2"
#	let "tid = ($k-211)/2"
#	echo $optimizer
#	echo $tid
#	if [ $optimizer -eq 0 ]; then
#		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-east1-b --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds\" \"$dataseeds\"" 
#	fi
#	if [ $optimizer -eq 1 ]; then
#		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-east1-b --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds2\" \"$dataseeds\"" 
#	fi
#
#done
#wait

