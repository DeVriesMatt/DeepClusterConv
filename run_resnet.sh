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

for layers in "[1,1,1,1]" "[2,2,2,2]" "[3,4,6,3]" "[3,4,23,3]" "[3,8,36,3]" "[3,24,36,3]"
do
        python main.py --net_architecture ResNet --output_dir '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/' --dataset_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/OPM_Roi_Images_Full_646464_Cluster3/OPM_Roi_Images_Full_646464_Cluster3' --train_lightning False --num_gpus 1 --num_features 20 --num_clusters 3 --resnet_layers $layers
done

