#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-2 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)
tids=(2073  2074  2076  2077  18  21  22  23  24  26  28  3020  3481  30  31  32  3505  3506  3507  36  3524  3526  3527  3530  3533  3  3536  43  45  58  3574  6)

seeds="1 5001"
#seeds1="10001 15001 20001 25001 30001 35001 40001 45001"
seeds2="50001 55001"
#seeds2="60001 65001 70001 75001 80001 85001 90001 95001"
#seeds3="100001 105001 110001 115001 120001 125001 130001 135001 140001 145001"
#seeds4="150001 155001 160001 165001 170001 175001 180001 185001 190001 195001"
dataseeds="1 1"
for k in $(seq 1 256); do 
	let "optimizer=($k-1)%8"
	let "tid = ($k-1)/8"
	echo $optimizer
	echo $tid
	if [ $optimizer -eq 0 ]; then
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/hyperopt ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" 
	elif [ $optimizer -eq 1 ]; then
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh smac/smac ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" 
	elif [ $optimizer -eq 2 ]; then
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh nvb_hyperband/nvb_hyperband ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" 
	elif [ $optimizer -eq 3 ]; then
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" 
	elif [ $optimizer -eq 4 ]; then
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh smac/smacnoinit ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" 
	elif [ $optimizer -eq 5 ]; then
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/hyperoptnoinit ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" 
	elif [ $optimizer -eq 6 ]; then
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh nvb_hyperband/nvb_hyperbandadap ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" 
	else
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds2\" \"$dataseeds\" 3600" 
	fi

done
wait

