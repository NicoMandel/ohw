#!/bin/bash
# nodes, tasks, cores, memory, GPU, GPU count
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH --partition=GPU
#SBATCH --gpus-per-node=1
# max job length (D-HH:MM), name, log locations
#SBATCH -t 0-20:51
#SBATCH --job-name=yoloTrain
#SBATCH --err=/mnt/scratch_lustre/<your_scratch_dir>/jobLogs/job-%j.err
#SBATCH --output/mnt/scratch_lustre/<your_scratch_dir>/jobLogs/job-%j.out

# alternative for selecting specific node: #SBATCH --nodelist=<node-name>
module purge 
module load go singularity

# structure: --bind <host_directory>:<directory_in_container>
singularity exec --nv --bind /mnt/scratch_lustre/<your_scratch-dir>/<data_dir>:/home/ubunu/data/ \
        --bind /mnt/scratch_lustre/<your_scratch-dir>/saves:/home/ubuntu/saves \
        /mnt/scratch_lustre/<your_scratch-dir>/singularity/<container>.simg python3 -u train.py     # actual training container