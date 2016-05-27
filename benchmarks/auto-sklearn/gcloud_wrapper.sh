#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-1 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)
tids=(45  3573  3011  58  3574  6  3581  3584  3586  3588  3589  3593  3594  3600  3601  3603  3606  3607  3609  3617  3618  3627  3638  3662  3664  3668  3671  3672  3678  3681  3684  3686  3687  3688  3698  3702  3708  3710  3711  3712  3714  3730  3735  3745  3760  3764  3766  3767  3773  3775  3776  3777  3780  3786  3793  3797  3816  3821  3822  3825  3829  3834  3839  3840  3841  3842  3843  3854  3856  3858)
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
	#	gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/hyperopt ${tids[$tid]} \"$seeds\" \"$dataseeds\"" 
	#fi
	#if [ $optimizer -eq 1 ]; then
	#	gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh smac/smac ${tids[$tid]} \"$seeds\" \"$dataseeds\"" 
	#fi
	if [ $optimizer -eq 2 ]; then
		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh nvb_hyperband/nvb_hyperband ${tids[$tid]} \"$seeds\" \"$dataseeds\"" 
	fi

done
wait

#for k in $(seq 211 350); do 
#	let "optimizer=($k-211)%2"
#	let "tid = ($k-211)/2"
#	echo $optimizer
#	echo $tid
#	if [ $optimizer -eq 0 ]; then
#		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds\" \"$dataseeds\"" 
#	fi
#	if [ $optimizer -eq 1 ]; then
#		gcloud compute ssh lisha_c_li@${instances_list[$k]} --zone us-central1-f --command "cd /home/lisha_c_li; screen -dm ./experiment_loop.sh tpe/random_hyperopt ${tids[$tid]} \"$seeds2\" \"$dataseeds\"" 
#	fi
#
#done
#wait

