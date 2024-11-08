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
#SBATCH --err=/mnt/scratch_lustre/hawkweed_drone_scratch/log/job-%j.err
#SBATCH --output/mnt/scratch_lustre/hawkweed_drone_scratch/log/job-%j.out

# alternative for selecting specific node: #SBATCH --nodelist=<node-name>
module purge 
module load go singularity

# structure: --bind <host_directory>:<directory_in_container>
singularity exec --nv --bind /mnt/scratch_lustre/hawkweed_drone_scratch/data:/home/ubuntu/datasets/ \
        --bind /home/hathenbd/scripts/hawkweed/ohw/scripts:/home/ubuntu/ \
        --bind /home/hathenbd/scripts/hawkweed/ohw/src/ohw:/home/ubuntu/ohw \
        --bind /home/hathenbd/scripts/hawkweed/ohw/data:/home/ubuntu/data \
        --bind /mnt/scratch_lustre/hawkweed_drone_scratch/saves:/home/ubuntu/results \
        /mnt/scratch_lustre/hawkweed_drone_scratch/singularity/<container>.simg python3 -u train.py     # actual training container