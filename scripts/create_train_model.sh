#!/bin/bash

# code and data reposiories
code_repo="/home/mandeln/ohw"
base_shared_dir="/mnt/scratch_lustre/hawkweed_drone_scratch"

# training both datasets
datasets=("1cm" "024cm")

# training multiple model sizes
model_size=("n" "s" "m")

module load slurm
# echo all the relevant factors into the base file by structure
for ms in "${model_size[@]}"; do
    for ds in "${datasets[@]}"; do
        # create job name
        jn="$ms-$ds"

        # automatically fill in sbatch file
        echo "#!/bin/bash" >> "$jn".sh
        echo "#SBATCH -N 1" >> "$jn".sh
        echo "#SBATCH -n 1" >> "$jn".sh
        echo "#SBATCH -c 8" >> "$jn".sh
        echo "#SBATCH --mem 32G" >> "$jn".sh
        echo "#SBATCH --partition=GPU" >> "$jn".sh
        echo "#SBATCH --gpus-per-node=1" >> "$jn".sh
        echo "#SBATCH --mem 32G" >> "$jn".sh
        echo "#SBATCH -t 0-10:59" >> "$jn".sh
        echo "#SBATCH --job-name=\"$jn\"" >> "$jn".sh
        echo "#SBATCH --err=$base_shared_dir/log_nico/training/job-%j.err" >> "$jn".sh
        echo "#SBATCH --output=$base_shared_dir/log_nico/training/job-%j.out" >> "$jn".sh

        # module parts
        echo "module purge" >> "$jn".sh
        echo "module load go singularity" >> "$jn".sh

        # actual job - ensure that the directories are correct - input and output!
        echo "singularity exec --nv --pwd /home/ubuntu --bind $code_repo/scripts:/home/ubuntu/ \
            --bind $base_shared_dir/data_nico:/home/ubuntu/datasets \
            --bind $code_repo/src/ohw:/home/ubuntu/ohw \
            --bind $base_shared_dir/results_nico:/home/ubuntu/results \
            $base_shared_dir/pt-sahi-123.simg python3 -u train_model.py \
            $ms datasets/$ds/$ds.yaml --save results/model_res.xlsx -b 8" >> "$jn".sh

        # choose between sbatch "$jn.sh" or cat "$jn.sh"
        sbatch "$jn.sh"
        # cat "$jn".sh
        rm "$jn".sh
    done
done