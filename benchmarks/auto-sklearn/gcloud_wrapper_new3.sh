#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-3 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)
tids=(3581  3584  3586  3588  3589  3593  3594  3600  3601  3603  3606  3607  3609  3617  3618  3627  3638  3662  3664  3668  3671  3672  3678  3681  3684  3686  3687  3688  3698  3702  3708  3710)

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

