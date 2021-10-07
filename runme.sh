#!/bin/bash
train_num=$1

func(){
    mkdir -p $3
    
    echo "nohup python train.py \
	--task_id $2 \
	--save_dir $3 \
	--num_episodes $1" > $3/$2.log

    nohup python train_lstm.py \
	    --task_id $2 \
	    --save_dir $3 \
	    --num_episodes $1 > $3/train.log &
}	

for((i=0;i<$train_num;i++))
do
    export CUDA_VISIBLE_DEVICES=$(($i%8))
    echo "start tack ${i}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}}"
    func $2 ${i} train_dir_rp${i}
    sleep 0.1
done

