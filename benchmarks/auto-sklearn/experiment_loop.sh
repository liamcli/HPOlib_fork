#!/bin/bash

#First argument is path to optimizer
#Second argument is openml_tid
#Third argument are seeds
seeds=("$3")
export PYTHONPATH=/home/lisha_c_li/HPOlib/
#cd /home/lisha_c_li/HPOlib/benchmarks/auto-sklearn/nocv
for i in $seeds
do sudo PYTHONPATH=$PYTHONPATH HPOlib-run -o ../../../optimizers/$1 -s $i --HPOLIB:experiment_directory_prefix open_ml_$2_ --EXPERIMENT:openml_tid $2 
done
