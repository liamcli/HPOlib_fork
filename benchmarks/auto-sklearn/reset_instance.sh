output=$(gcloud compute instance-groups managed list-instances instance-group-1 --zone us-central1-f | cut -d' ' -f1)
instances_list=($output)

for k in $(seq 1 185); do 
gcloud compute instances reset  ${instances_list[$k]} --zone us-central1-f &
done
wait
