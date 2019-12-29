#!/bin/bash

#SBATCH --ntasks-per-node=12
#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH -p bigmem
#SBATCH --mail-user=aubry.d-andoque-de-seriege@student.isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=Dict_building
#SBATCH -o ISAE_Comp/log/slurm.%j.out # STDOUT
#SBATCH -e ISAE_Comp/log/slurm.%j.err # STDERR

conda activate myenv
python3 /scratch/students/a.d-andoque/INF554-LinkPredictionProject/ISAE_Comp/Build_dict_and_bow.py