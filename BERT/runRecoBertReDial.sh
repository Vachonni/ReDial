#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --mail-user=nicholasvachon@hotmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=10                                                             # Ask for 4 CPUs
#SBATCH --gres=gpu:4                                                                   # Ask for 1 GPU. If want specific -> gres=gpu:titanx:1
#SBATCH --mem=16G                                                                      # Ask for xx GB of RAM
#SBATCH --time=0-05:00                                                                 # The job will run for DD-HH:MM:SS 
#SBATCH -o /home/vachonni/scratch/ReDial_A19/slurm_RecoText_SR_Reco_ratings_ReDOrId-%j.out      # Write the log on tmp1



# ID of my experience 
DATA_TYPE="RecoText_SR_Reco_ratings_ReDOrId"
EXP_TYPE="RecoText_SR_Reco_ratings_ReDOrId"




# 1. Load your environment
module load python/3.7 scipy-stack && source ~/ENV/ENV_ReDial_A19/bin/activate      

# 2. Copy your dataset on the compute node
cp -r /home/vachonni/scratch/ReDial_A19/Data/ReDial/$DATA_TYPE/* $SLURM_TMPDIR

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python BertForReco.py --log_path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --epoch 30

# 4. Copy whatever you want to save on $SCRATCH after creating the directory if doesn't exist
mkdir -p /home/vachonni/scratch/ReDial_A19/Results/$EXP_TYPE/
cp -r $SLURM_TMPDIR/model_out/  /home/vachonni/scratch/ReDial_A19/Results/$EXP_TYPE/
#cp /home/vachonni/scratch/ReDial_A19/slurm-%j.out /home/vachonni/scratch/ReDial_A19/Results/$EXP_TYPE/

