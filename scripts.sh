#!/bin/bash

#SBATCH --job-name=ego4d_MQ_test
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=100000

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

python Train.py --is_train true --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --project_name Ego4d_final2 --use_ReMoT --train_lr 0.0001 --num_epoch 1 --num_heads 8 --mask_size 32 --num_levels 6 --run_name ReMoTlarge --bb_hidden_dim 512 --dim_attention 512 --mlp_num_hiddens 4096

python Infer.py --is_train false --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --output_path /data/s5091217/Ego4d-main/ego4d_output --project_name Ego4d_final2 --use_ReMoT --train_lr 0.0001 --num_epoch 1 --num_heads 8 --mask_size 32 --num_levels 6 --run_name ReMoTlarge --bb_hidden_dim 512 --dim_attention 512 --mlp_num_hiddens 4096

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --project_name Ego4d_final2 --use_ReMoT --train_lr 0.0001 --num_epoch 1 --num_heads 8 --mask_size 32 --num_levels 6 --run_name ReMoTlarge --bb_hidden_dim 512 --dim_attention 512 --mlp_num_hiddens 4096



python Train.py --is_train true --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --project_name Ego4d_final2 --run_name xGPN --use_xGPN --num_epoch 1

python Infer.py --is_train false --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --output_path /data/s5091217/Ego4d-main/ego4d_output --project_name Ego4d_final2 --run_name xGPN --use_xGPN

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --project_name Ego4d_final2 --run_name xGPN --use_xGPN --num_epoch 1



python Train.py --is_train true --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --project_name Ego4d_final2 --run_name vanila --num_epoch 1

python Infer.py --is_train false --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --output_path /data/s5091217/Ego4d-main/ego4d_output --project_name Ego4d_final2 --run_name vanila --num_epoch 1

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --project_name Ego4d_final2 --run_name vanila --num_epoch 1



python Train.py --is_train true --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --project_name Ego4d_final2 --run_name ViT2 --use_ViT2 --train_lr 0.0001 --num_epoch 1 --num_heads 8 --mask_size 32 --num_levels 6 --bb_hidden_dim 256 --dim_attention 256 --mlp_num_hiddens 2048

python Infer.py --is_train false --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --output_path /data/s5091217/Ego4d-main/ego4d_output --project_name Ego4d_final2 --run_name ViT2 --use_ViT2 --train_lr 0.0001 --num_epoch 1 --num_heads 8 --mask_size 32 --num_levels 6 --bb_hidden_dim 256 --dim_attention 256 --mlp_num_hiddens 2048

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --project_name Ego4d_final2 --run_name ViT2 --use_ViT2 --train_lr 0.0001 --num_epoch 1 --num_heads 8 --mask_size 32 --num_levels 6 --bb_hidden_dim 256 --dim_attention 256 --mlp_num_hiddens 2048



python Train.py --is_train true --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --project_name Ego4d_final2 --use_ReMoT --train_lr 0.0001 --num_epoch 1 --num_heads 8 --mask_size 32 --num_levels 6 --run_name ReMoTsmall --bb_hidden_dim 256 --dim_attention 256 --mlp_num_hiddens 2048

python Infer.py --is_train false --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --output_path /data/s5091217/Ego4d-main/ego4d_output --project_name Ego4d_final2 --use_ReMoT --train_lr 0.0001 --num_epoch 1 --num_heads 8 --mask_size 32 --num_levels 6 --run_name ReMoTsmall --bb_hidden_dim 256 --dim_attention 256 --mlp_num_hiddens 2048

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all --project_name Ego4d_final2 --use_ReMoT --train_lr 0.0001 --num_epoch 1 --num_heads 8 --mask_size 32 --num_levels 6 --run_name ReMoTsmall --bb_hidden_dim 256 --dim_attention 256 --mlp_num_hiddens 2048

