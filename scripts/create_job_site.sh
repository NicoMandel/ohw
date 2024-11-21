#!/bin/bash

# Start by changing this
site_location="~/src/csu/OHW_data/SDC_Sites/2312122_PP"
# site_location="/home/nico/src/csu/OHW_data/SDC_Sites/2312122_PP"
# automated job naming and finding subdirectories
sitename=$(basename "$site_location")
mapfile -t flightdirs < <(find "$site_location" -maxdepth 1 -type d -iname "flight*")

# echo all the relevant factors into the base file by structure
for jobsite in "${flightdirs[@]}"; do
    # create job name
    jobname=$(basename "$jobsite")
    jn="$sitename-$jobname"

    # automatically fill in sbatch file
    echo "#!/bin/bash" >> "$jn".sh
    echo "#SBATCH -N 1" >> "$jn".sh
    echo "#SBATCH -n 1" >> "$jn".sh
    echo "#SBATCH -c 4" >> "$jn".sh
    echo "#SBATCH --mem 32G" >> "$jn".sh
    echo "#SBATCH --partition=GPU" >> "$jn".sh
    echo "#SBATCH --gpus-per-node=1" >> "$jn".sh
    echo "#SBATCH --mem 32G" >> "$jn".sh
    echo "#SBATCH -t 0-03:59" >> "$jn".sh
    echo "#SBATCH --job-name=$jn" >> "$jn".sh
    echo "#SBATCH --err=/mnt/scratch_lustre/hawkweed_drone_scratch/log_nico-job-%j.err" >> "$jn".sh
    echo "#SBATCH --output=/mnt/scratch_lustre/hawkweed_drone_scratch/log_nico/job-%j.out" >> "$jn".sh

    # module parts
    echo "module purge" >> "$jn".sh
    echo "module load go singularity" >> "$jn".sh

    # actual job - ensure that the directories are correct - input and output!
    echo "singularity exec --nv --pwd /home/ubuntu --bind /home/mandeln/ohw/scripts:/home/ubuntu/ \
            --bind /home/mandeln/ohw/src/ohw:/home/ubuntu/ohw \
            --bind /mnt/scratch_lustre/hawkweed_drone_scratch/data_nico/inference:/home/ubuntu/inference \
            --bind /mnt/scratch_lustre/hawkweed_drone_scratch/results_nico:/home/ubuntu/results \
            /mnt/scratch_lustre/hawkweed_drone_scratch/yolo-rawpy.simg python3 -u inference_sahi.py \
            $jobsite results/20241109-n-1cm-1cm/weights/best.pt inference -n \"$sitename/$jobname\" -s -v" >> "$jn".sh

    # choose between sbatch "$jn.sh" or cat "$jn.sh"
    sbatch "$jn.sh"
    # cat "$jn".sh
    rm "$jn".sh
done