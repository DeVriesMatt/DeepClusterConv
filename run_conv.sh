#!/bin/bash
#SBATCH --job-name=deep_cluster_conv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=24:00:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/output%j.out
#SBATCH --error=/home/mvries/DeepClusterConv/error_file/error%j.err
#SBATCH --partition=gpu
module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh

conda activate dcfn


python main.py --net_architecture 'CAE_bn3' --output_dir '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/' --dataset_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/Datasets/VickyPlates/Treatments_plate_002_166464' --train_lightning False --num_gpus 1 --num_features 20 --num_clusters 10 --rate_pretrain 0.0002 --rate 0.0000002 --pretrain False --pretrained_net '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/nets/CAE_bn3_004_pretrained.pt'


