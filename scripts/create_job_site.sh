#!/bin/bash

# code and data reposiories
code_repo="/home/mandeln/ohw"
base_shared_dir="/mnt/scratch_lustre/hawkweed_drone_scratch"

# Start by changing this
site_location="/mnt/load/aerial/drone_uav/2023-24/NPWS_OHW_4/23.12.18 - Yarrabee North"
# site_location="/home/nico/src/csu/OHW_data/SDC_Sites/2312122_PP"
resolution="024cm" # alternative -> "1cm"
model_registry="results/model_res.xlsx"

# automated job naming and finding subdirectories
sitename=$(basename "$site_location")
mapfile -t flightdirs < <(find "$site_location" -maxdepth 1 -type d -iname "flight*")

module load slurm
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
    echo "#SBATCH -t 0-01:59" >> "$jn".sh           
    echo "#SBATCH --job-name=\"$jn\"" >> "$jn".sh
    echo "#SBATCH --err=$base_shared_dir/log_nico/inference/job-%j.err" >> "$jn".sh
    echo "#SBATCH --output=$base_shared_dir/log_nico/inference/job-%j.out" >> "$jn".sh

    # module parts
    echo "module purge" >> "$jn".sh
    echo "module load go singularity" >> "$jn".sh

    # actual job - ensure that the directories are correct - input and output!
    echo "singularity exec --nv --pwd /home/ubuntu --bind $code_repo/scripts:/home/ubuntu/ \
            --bind $code_repo/src/ohw:/home/ubuntu/ohw \
            --bind \"$jobsite\":/home/ubuntu/inference \
            --bind $base_shared_dir/inference_out:/home/ubuntu/inference_out \
            --bind $base_shared_dir/results_nico:/home/ubuntu/results \
            $base_shared_dir/pt-sahi-123.simg python3 -u inference_sahi.py \
            inference $model_registry $resolution inference_out -n \"$sitename/$jobname\" -s -v" >> "$jn".sh

    # choose between sbatch "$jn.sh" or cat "$jn.sh"
    sbatch "$jn.sh"
    # cat "$jn".sh
    rm "$jn".sh
done