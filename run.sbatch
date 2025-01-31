#!/bin/bash

#SBATCH -J euijin-planet                    # Job 이름
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH -o /home/euijinrnd/sbatch_log/corl_exp2_0.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1                      # Job에 사용할 리소스 (GPU)
python main_planet.py --configs defaults dmc
$@
