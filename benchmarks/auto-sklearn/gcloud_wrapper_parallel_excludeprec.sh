#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-1 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)
tids=(3881  3882  3889  3893  3904  3907  3953  3954  2071  2072  2076  24  26  3481  32  6  3588  3593  3600  3601  3618  3627  3672  3681  3684  3686  3687  3688  3698  3708  3711  3745  3764  3786  3822  3839 3840)

#seeds="25001 30001"
seeds1="1 5001 10001 15001 20001 25001 30001 35001 40001 45001"
#seeds2="75001 80001"
seeds2="50001 55001 60001 65001 70001 75001 80001 85001 90001 95001"
#seeds3="100001 105001 110001 115001 120001 125001 130001 135001 140001 145001"
#seeds4="150001 155001 160001 165001 170001 175001 180001 185001 190001 195001"
dataseeds="1 1 1 1 1 1 1 1 1 1"
let "k=0"
for tid in $(seq 0 36); do 
	let "k1=$k+1"
	let "k2=$k+2"
	let "k3=$k+3"
	let "k4=$k+4"
	let "k5=$k+5"
	#let "k6=$k+6"
	#let "k7=$k+7"
	#let "k8=$k+8"
	#gcloud compute ssh lisha_c_li@${instances_list[$k1]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/hyperopt ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" &
	#gcloud compute ssh lisha_c_li@${instances_list[$k2]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh smac/smac ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k1]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh nvb_hyperband/nvb_hyperband ${tids[$tid]} \"$seeds1\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k2]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds1\" \"$dataseeds\" 3600" &
	#gcloud compute ssh lisha_c_li@${instances_list[$k2]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh smac/smacnoinit ${tids[$tid]} \"$seeds\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k3]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/hyperoptnoinit ${tids[$tid]} \"$seeds1\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k4]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh nvb_hyperband/nvb_hyperbandadap ${tids[$tid]} \"$seeds1\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k5]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds2\" \"$dataseeds\" 3600" &
	#echo $tid
	#echo $k1 &
	#echo $k2 &
	#echo $k3 &
	#echo $k4 &
	#echo $k5 &
	#echo $k6 &
	#echo $k7 &
	#echo $k8 &
	let "k=$k1+4"
wait
done
wait

