#!/bin/bash

# echo all the relevant factors into the base file by structure
jobsite="testfile"
echo "#!/bin/bash" >> "$jobsite".sh
echo "#SBATCH -N 1" >> "$jobsite".sh
echo "#SBATCH -n 1" >> "$jobsite".sh
echo "#SBATCH -c 4" >> "$jobsite".sh
echo "#SBATCH --mem 32G" >> "$jobsite".sh
echo "#SBATCH --partition=GPU" >> "$jobsite".sh
echo "#SBATCH --gpus-per-node=1" >> "$jobsite".sh
echo "#SBATCH --mem 32G" >> "$jobsite".sh
echo "#SBATCH -t 0-03:59" >> "$jobsite".sh
echo "#SBATCH --job-name=$jobsite" >> "$jobsite".sh
echo "#SBATCH --err=/mnt/scratch_lustre/hawkweed_drone_scratch/log_nico-job-%j.err" >> "$jobsite".sh
echo "#SBATCH --output=/mnt/scratch_lustre/hawkweed_drone_scratch/log_nico/job-%j.out" >> "$jobsite".sh

# module parts
echo "module purge" >> "$jobsite".sh
echo "module load go singularity" >> "$jobsite".sh

# actual job
echo "singularity exec --nv --pwd /home/ubuntu --bind /home/mandeln/ohw/scripts:/home/ubuntu/ \
        --bind /home/mandeln/ohw/src/ohw:/home/ubuntu/ohw \
        --bind /mnt/scratch_lustre/hawkweed_drone_scratch/data_nico/inference:/home/ubuntu/inference \
        --bind /mnt/scratch_lustre/hawkweed_drone_scratch/results_nico:/home/ubuntu/results \
        /mnt/scratch_lustre/hawkweed_drone_scratch/yolo-rawpy.simg python3 -u inference_sahi.py \
        $jobsite results/20241109-n-1cm-1cm/weights/best.pt inference -n singularity_1cm_test -s -v 
        " >> "$jobsite".sh

# replace with "sbatch $jobsite.sh"
cat "$jobsite".sh