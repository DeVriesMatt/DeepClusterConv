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

for architecture in 'CAE_3' 'CAE_bn3' 'CAE_4' 'CAE_bn4' 'CAE_5' 'CAE_bn5'
do
	python main.py --net_architecture $architecture --output_dir '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/' --dataset_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/OPM_Roi_Images_Full_646464_Cluster3/OPM_Roi_Images_Full_646464_Cluster3' --train_lightning False --num_gpus 1 --num_features 20 --num_clusters 3
done

