#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-1 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)
tids=(3865  3881  3882  3883  3884  3889  3893  3894  3902  3903  3904  3907  3917  3918  3919  3950  3953  3954  3962  3964  3968  3972  3973  3976  3980  3995  4000  12  14  16  2071  2072  2073  2074  2076  2077  18  21  22  23  24  26  28  3481  30  31  32  36  3  43  45  58  3574  6  3581  3584  3586  3588  3589)

seeds3="100001 105001 110001 115001 120001 125001 130001 135001 140001 145001"
seeds4="150001 155001 160001 165001 170001 175001 180001 185001 190001 195001"
dataseeds="1 1 1 1 1 1 1 1 1 1"
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
	gcloud compute ssh lisha_c_li@${instances_list[$k1]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/hyperopt ${tids[$tid]} \"$seeds3\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k2]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh smac/smac ${tids[$tid]} \"$seeds3\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k3]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh nvb_hyperband/nvb_hyperband ${tids[$tid]} \"$seeds3\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k4]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds3\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k5]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh smac/smacnoinit ${tids[$tid]} \"$seeds3\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k6]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/hyperoptnoinit ${tids[$tid]} \"$seeds3\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k7]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh nvb_hyperband/nvb_hyperbandadap ${tids[$tid]} \"$seeds3\" \"$dataseeds\" 3600" &
	gcloud compute ssh lisha_c_li@${instances_list[$k8]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds4\" \"$dataseeds\" 3600" &
	#echo $tid
	#echo $k1 &
	#echo $k2 &
	#echo $k3 &
	#echo $k4 &
	#echo $k5 &
	#echo $k6 &
	#echo $k7 &
	#echo $k8 &
	let "k=$k1+7"
wait
done
wait

