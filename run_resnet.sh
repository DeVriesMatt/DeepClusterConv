#!/bin/bash
#SBATCH --job-name=deep_cluster_conv_resnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=96:00:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/output%j.out
#SBATCH --error=/home/mvries/DeepClusterConv/error_file/error%j.err
#SBATCH --partition=gpu
module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh

conda activate dcfn

python main.py  --epochs 500 --epochs_pretrain 200 --num_features 50 --output_dir '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/' --dataset_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/Datasets/SingleCell_ERK_Cell_128/' --custom_img_size 128 --update_interval 1 --gamma 0.999 --dataset 'SingleCellERK_128' --batch_size 8 --net_architecture 'ResNet' --pretrain True 

