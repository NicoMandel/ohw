#!/bin/bash

# code and data reposiories
code_repo="/mnt/appsource/local/hawkweed_drone/ohw"
base_shared_dir="/mnt/scratch_lustre/hawkweed_drone_scratch"
dataset="" # alternative -> "1cm"
model_size="m"
registry=""

usage() {
    printf "\nUsage : $0 -d <dataset>
    Options:
        -d dataset to be used. Required. Choose from any in $base_share_dir/data 
        -m model size to be used. Choose from n,s,m,l. defaults to m 
        -r Model registry file for storing results. If not given, will just save. Advised to set to $base_shared_dir/results/model_res.xlsx
        -h display this help message
        
        Example:
            $0 -d 'dataset' -m 's' -r '/path/to/registry.xlsx'
"
}

# argument parsing
while getopts 'd:m:r:h' flag; do 
    case "${flag}" in
        d) dataset="${OPTARG}" ;;
        m) model_size="${OPTARG}" ;;
        r) registry="${OPTARG}" ;;
        h) usage 
            exit -1;;
        *) usage
            exit -1;;
    esac
done

# check required arguments
if [ -z "$dataset" ]; then
    echo "ERROR: Dataset is required!".
    usage
    exit -1
fi


# set registry string if given
if [[ -n "$registry" ]]; then
    registry_string="--save $registry"
else
    registry_string=""
fi

# creating log diretory
mkdir -p "$base_shared_dir/log/training/$dataset/$model_size"

# echo all the relevant factors into the base file by structure
# create job name
jn="$dataset-$model_size"

# automatically fill in sbatch file
echo "#!/bin/bash" >> "$jn".sh
echo "#SBATCH -N 1" >> "$jn".sh
echo "#SBATCH -n 1" >> "$jn".sh
echo "#SBATCH -c 8" >> "$jn".sh
echo "#SBATCH --mem 32G" >> "$jn".sh
echo "#SBATCH --partition=GPU" >> "$jn".sh
echo "#SBATCH --gpus-per-node=1" >> "$jn".sh
echo "#SBATCH -t 0-10:59" >> "$jn".sh
echo "#SBATCH --job-name=\"$jn\"" >> "$jn".sh
echo "#SBATCH --err=$base_shared_dir/log/training/$dataset/$model_size/job-%j.err" >> "$jn".sh
echo "#SBATCH --output=$base_shared_dir/log/training/$dataset/$model_size/job-%j.out" >> "$jn".sh

# module parts
echo "module purge" >> "$jn".sh
echo "module load go singularity" >> "$jn".sh

# Environment variables
echo "export SINGULARITYENV_RANK=-1" >> "$jn".sh 
echo "export SINGULARITYENV_LOCAL_RANK=-1" >> "$jn".sh

# actual job - ensure that the directories are correct - input and output!
echo "singularity exec --nv --pwd /home/ubuntu \
    --bind $code_repo/scripts:/home/ubuntu/ \
    --bind $base_shared_dir/data:/home/ubuntu/datasets \
    --bind $code_repo/src/ohw:/home/ubuntu/ohw \
    --bind $base_shared_dir/results_2025:/home/ubuntu/results \
    $base_shared_dir/pt-ul-8281.simg python3 -u train_model.py \
    $model_size datasets/$ds/$ds.yaml $registry_string" >> "$jn".sh

# choose between sbatch "$jn.sh" or cat "$jn.sh"
sbatch "$jn.sh"
# cat "$jn".sh
rm "$jn".sh