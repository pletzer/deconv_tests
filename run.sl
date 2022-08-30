#!/bin/bash -e
#SBATCH --job-name=test_fftw_3d_r2c # job name (shows up in the queue)
#SBATCH --time=00:10:00      # Walltime (HH:MM:SS)
#SBATCH --mem=4g          # Memory in MB

srun test_fftw_3d_r2c $1 40
