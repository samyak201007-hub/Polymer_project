#!/bin/bash
#SBATCH --job-name=polymer_ML
#SBATCH --partition=phd_student
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=polymer_sim_%j.log

module load lammps-openmpi

echo "Starting LAMMPS job..."

mpirun -np 4 lmp -in polymer_coil_globule.in

echo "LAMMPS job finished! Check the shape_data.txt file for your results."
