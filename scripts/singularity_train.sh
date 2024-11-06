#!/bin/bash
basedir="/media/nico/sdcReplica/"
# structure: --bind <host_directory>:<directory_in_container>
# volumes to bind:
        # datasets -> actual data files
        # src/ohw -> ohw
        # 
        
singularity exec --nv --bind /mnt/scratch_lustre/hawkweed_drone_scratch/data:/home/ubuntu/datasets/ \
        --bind /home/hathenbd/scripts/hawkweed/ohw/scripts:/home/ubuntu/ \
        --bind /home/hathenbd/scripts/hawkweed/ohw/src/ohw:/home/ubuntu/ohw \
        --bind /home/hathenbd/scripts/hawkweed/ohw/data:/home/ubuntu/data \
        --bind /mnt/scratch_lustre/hawkweed_drone_scratch/saves:/home/ubuntu/results \
        "$basedir/scratch_lustre/yolo.simg" python3 -u train.py     # actual training container