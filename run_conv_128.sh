#!/bin/bash
#SBATCH --job-name=deep_cluster_conv_128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=250:00:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/%joutput.out
#SBATCH --error=/home/mvries/DeepClusterConv/error_file/%jerror.err
#SBATCH --partition=gpu
module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh

conda activate dcfn

for num_features in 10 50 100 200 512
do
    python main.py  --epochs 500 --epochs_pretrain 300 --num_features $num_features --output_dir '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/DeepClusterConv/' --dataset_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/Datasets/SinglecellERK_128/' --update_interval 1 --gamma 0.999 --dataset 'SinglecellERK_128'
done
