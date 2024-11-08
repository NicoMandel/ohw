#!/bin/bash
basedir="/media/nico/sdcReplica"
# structure: --bind <host_directory>:<directory_in_container>
# volumes to bind:
        # datasets -> actual data files
        # src/ohw -> ohw
        # 
        
# singularity exec --nv --bind /mnt/scratch_lustre/hawkweed_drone_scratch/data:/home/ubuntu/datasets/ \
#         --bind /home/hathenbd/scripts/hawkweed/ohw/scripts:/home/ubuntu/ \
#         --bind /home/hathenbd/scripts/hawkweed/ohw/src/ohw:/home/ubuntu/ohw \
#         --bind /home/hathenbd/scripts/hawkweed/ohw/data:/home/ubuntu/data \
#         --bind /mnt/scratch_lustre/hawkweed_drone_scratch/saves:/home/ubuntu/results \
#         "$basedir/scratch_lustre/yolo.simg" python3 -u train.py     # actual training container

singularity shell --bind "$basedir/appsource/local/hawkweed_drone/ohw/scripts":/home/ubuntu/ \
        --bind "$basedir/appsource/local/hawkweed_drone/ohw/src/ohw":/home/ubuntu/ohw \
        --bind "$basedir/scratch_lustre/datasets/base":/home/ubuntu/datasets \
        --bind "$basedir/scratch_lustre/results/":/home/ubuntu/results \
        "$basedir/scratch_lustre/yolo-rawpy.simg"