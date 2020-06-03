#!/bin/bash

ROOT_DIR=/home/robert/PycharmProjects

# echo "* Create dataset"
# python3 scripts/create_dataset.py \
# 	--src_dir $ROOT_DIR/upb_dataset \
# 	--dst_dir dataset \
# 	--split_dir $ROOT_DIR/disertatie/scenes_split \
# 	--dataset upb


echo "* Training"
python3 train.py \
	--dataset upb \
	--batch_size 12 \
	--num_epochs 20 \
	--scheduler_step_size 15 \
	--height 256 \
	--width 512 \

