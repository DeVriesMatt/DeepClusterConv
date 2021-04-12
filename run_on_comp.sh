#!/usr/bin/env sh

for num_features in 10 50 100 200 512
do
  python main.py  --epochs 1000 --epochs_pretrain 300 --num_features $num_features --update_interval 1
done

