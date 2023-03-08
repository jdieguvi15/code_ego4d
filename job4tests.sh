#!/bin/bash

#SBATCH --job-name=ego4d_MQ_test
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=10000

pwd
module purge
module add Anaconda3
module add GCC
module list
conda activate d2l
conda activate pytorch160

cd /data/s5091217/code_ego4d/
git pull

#es importante que todos tengan los mismo parámetros
cd /data/s5091217/code_ego4d/episodic-memory-main/MQ

python Train.py --is_train true --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features s --clip_anno Evaluation/ego4d/annot/clip_ann_v1.json

python Infer.py --is_train false --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models  --output_path /data/s5091217/Ego4d-main/ego4d_output --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features s --clip_anno Evaluation/ego4d/annot/clip_ann_v1.json

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features s --clip_anno Evaluation/ego4d/annot/clip_ann_v1.json

python Train.py --is_train true --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features s --clip_anno Evaluation/ego4d/annot/clip_ann_v2.json

python Infer.py --is_train false --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models  --output_path /data/s5091217/Ego4d-main/ego4d_output --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features s --clip_anno Evaluation/ego4d/annot/clip_ann_v2.json

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features s --clip_anno Evaluation/ego4d/annot/clip_ann_v2.json

python Train.py --is_train true --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features o --clip_anno Evaluation/ego4d/annot/clip_ann_v1.json

python Infer.py --is_train false --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models  --output_path /data/s5091217/Ego4d-main/ego4d_output --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features o --clip_anno Evaluation/ego4d/annot/clip_ann_v1.json

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features o --clip_anno Evaluation/ego4d/annot/clip_ann_v1.json

python Train.py --is_train true --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features o --clip_anno Evaluation/ego4d/annot/clip_ann_v2.json

python Infer.py --is_train false --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models  --output_path /data/s5091217/Ego4d-main/ego4d_output --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features o --clip_anno Evaluation/ego4d/annot/clip_ann_v2.json

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features o --clip_anno Evaluation/ego4d/annot/clip_ann_v2.json

python Train.py --is_train true --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features so --clip_anno Evaluation/ego4d/annot/clip_ann_v1.json

python Infer.py --is_train false --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models  --output_path /data/s5091217/Ego4d-main/ego4d_output --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features so --clip_anno Evaluation/ego4d/annot/clip_ann_v1.json

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features so --clip_anno Evaluation/ego4d/annot/clip_ann_v1.json

python Train.py --is_train true --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features so --clip_anno Evaluation/ego4d/annot/clip_ann_v2.json

python Infer.py --is_train false --dataset ego4d  --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models  --output_path /data/s5091217/Ego4d-main/ego4d_output --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features so --clip_anno Evaluation/ego4d/annot/clip_ann_v2.json

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --use_ReMoT --train_lr 0.0001 --num_epoch 30 --features so --clip_anno Evaluation/ego4d/annot/clip_ann_v2.json

