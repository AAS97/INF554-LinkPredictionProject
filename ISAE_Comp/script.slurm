#!/bin/bash

#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH --mail-user=aubry.d-andoque-de-seriege@student.isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=text_embedding
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR

conda activate myenv
python3 /scratch/students/a.d-andoque/INF554-LinkPredictionProject/ISAE_Comp/Text_Processing.py