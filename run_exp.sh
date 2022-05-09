## declare an array variable
declare -a arr=("smile" "mine" "infonce" "dv" "nwj")

## now loop through the above array
for E in "${arr[@]}"
do
   echo "$E"
   for M in 0
    do 
        echo "$M"
        python images.py --dataset-name bert \
        --masks "([$M],[$M])" \
        --learning-rate "1e-4" \
        --batch-size 64 --critic-net-name exp \
        --estimator-name $E \
        --fig-dir "experiments_bert" \
        --iterations 10000
    done
   # or do whatever with individual element of the array
done