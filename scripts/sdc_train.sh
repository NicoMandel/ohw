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
#SBATCH --err=/mnt/scratch_lustre/hawkweed_drone_scratch/log_nico/job-%j.err
#SBATCH --output=/mnt/scratch_lustre/hawkweed_drone_scratch/log_nico/job-%j.out

# alternative for selecting specific node: #SBATCH --nodelist=<node-name>
module purge 
module load go singularity

# structure: --bind <host_directory>:<directory_in_container>
singularity exec --nv --bind /mnt/scratch_lustre/hawkweed_drone_scratch/data_nico:/home/ubuntu/datasets/ \
        --bind /home/mandeln/ohw/scripts:/home/ubuntu/ \
        --bind /home/mandeln/ohw/src/ohw:/home/ubuntu/ohw \
        --bind /mnt/scratch_lustre/hawkweed_drone_scratch/results_nico:/home/ubuntu/results \
        /mnt/scratch_lustre/hawkweed_drone_scratch/yolo-rawpy.simg cd /home/ubuntu && python3 -u train_model.py \
        n datasets/1cm/1cm.yaml 
        # /home/ubuntu/ prefixed before files
        