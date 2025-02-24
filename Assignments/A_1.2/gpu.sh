#!/bin/sh
#SBATCH --job-name=phi3
#SBATCH --output vgg16.out
#SBATCH --error vgg16.err
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --gres=gpu:1
##SBATCH --mem=3000 


#SBATCH -p AI_Center_L40S 

##SBATCH -p v100-16gb-hiprio
##SBATCH -p v100-32gb-hiprio

##SBATCH -p gpu-v100-16gb
##SBATCH -p gpu-v100-32gb



##SBATCH -p OOD_gpu_32gb
##SBATCH -p dgx_aic



module load python3/anaconda/2023.1
module load cuda/12.1

# source activate /work/nayeem/ENV/llms_3 
source activate /work/nayeem/ENV/pytorchGPU_env

echo "date is: " date

echo "GPU: AI_Center_L40S"

##Add your code here:
echo " Hostname is:"
hostname

echo "Start time: $(date)"  # Captures the start time




python /work/nayeem/Neuromorphic/A_1.2/VGG_X.py



echo "End time: $(date)"  # Captures the start time



