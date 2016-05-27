#!/bin/bash
output=$(gcloud compute instance-groups managed list-instances instance-group-2 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)
tids=(3593  3594  3600  3601  3603  3606  3607  3609  3617  3618  3627  3638  3662  3664  3668  3671  3672  3678  3681  3684  3686  3687  3688  3698  3702  3708  3710  3711  3712  3714  3730  3735  3745  3760  3764  3766  3767  3773  3775  3776  3777  3780  3786  3793  3797  3816  3821  3822  3825  3829  3834  3839  3840  3841  3842  3843  3854  3856  3858)

#seeds3="100001 105001 110001 115001 120001 125001 130001 135001 140001 145001"
#seeds4="150001 155001 160001 165001 170001 175001 180001 185001 190001 195001"
seeds3="200001 205001 210001 215001 220001 225001 230001 235001 240001 245001"
seeds4="250001 255001 260001 265001 270001 275001 280001 285001 290001 295001"
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

