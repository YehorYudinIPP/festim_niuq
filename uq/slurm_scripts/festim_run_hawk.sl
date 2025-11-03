#!/bin/bash

### Job name
#SBATCH --job-name=festim_run_mpi_

### stdout and stderr
#SBATCH --output=slurm_logs/festim-run-out.%A_%a
#SBATCH --error=slurm_logs/festim-run-err.%A_%a

### Wallclock time (HOURS):MINUTES:SECONDS
#SBATCH --time=0:10:00

### Number of nodes and tasks
#SBATCH --nodes=1
#SBATCH --ntasks=4
##SBATCH --tasks-per-node=20
##SBATCH --ntasks-per-core=1
##SBATCH --cpus-per-tasks=8

### Hardware partition and quality of service
##SBATCH --partition=compute
##SBATCH --qos=

### Allocation
##SBATCH --account=
##SBATCH --mail-user=y.yudin@bangor.ac.uk

################

# Load modules
module load mpi/intel/2018/2
module load anaconda/2024.02

# Activate environments
#conda activate festim2-env
source activate festim2-env

# Prepare for the run
cd ../

# Diagnose before the run
echo "Using python from:" $(which python3)
echo "JobID: " ${SLURM_JOBID}
scontrol show job ${SLURM_JOBID}
echo "Number o SLURM tasks:" ${SLURM_NTASKS}

# Run the executable
## option 1) run via calling python interpreter
#python3 festim_model_run.py --config config/config.uq.yaml
## option 2) run via calling mpiexec
#mpiexec -n 16 python3 festim_model_run.py --config config/config.uq.yaml_par 
## option 3) run via calling mpirun
mpirun -np ${SLURM_NTASKS} python3 festim_model_run.py --config config/config.uq.yaml_par

# Postprocess

# Finish
echo "Finished FESTIM run!"

