#!/bin/bash
#SBATCH --job-name=deep_cluster_conv_shapenet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=96:00:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/%joutput.out
#SBATCH --error=/home/mvries/DeepClusterConv/error_file/%jerror.err
#SBATCH --partition=gpu
module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh

conda activate dcfn

for feats in 10 50 512 1024 2048
        do
                python main.py  --update_interval 1000 --epochs 3500 --epochs_pretrain 3500 --num_features $feats --output_dir '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/' --dataset_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/Datasets/ShapeNetVoxel/' --csv_dataset_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/Datasets/allData.csv'  --custom_img_size 64 --tol 0.000000001 --gamma 0.1 --dataset 'ShapeNetVoxel' --batch_size 32 --net_architecture 'CAE_bn3_Seq' --pretrain True --mode 'pretrain'
        done


