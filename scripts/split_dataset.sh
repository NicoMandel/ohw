#!/bin/bash

# code and data reposiories
code_repo="/mnt/appsource/local/hawkweed_drone/ohw"
base_shared_dir="/mnt/scratch_lustre/hawkweed_drone_scratch"

usage() {
    printf "\nUsage : $0 -i <input>
    Options:
        -i input folder of the dataset. Top level root folder, so above 'train' and 'test'
        -h display this help message
        
        Example:
            $0 -i '/path/to/folder' 
"
}

# default values
site_location=""

# argument parsing
while getopts 'i:h' flag; do 
    case "${flag}" in
        i) site_location="${OPTARG}" ;;
        h) usage 
            exit -1;;
        *) usage
            exit -1;;
    esac
done

# check required arguments
if [  ! -d "$site_location/train/" ]; then
    echo "ERROR: train subdirectory is required!"
    content=$(ls $site_location)
    echo "Content of $site_location is: $content"
    usage
    exit -1
fi

if [  ! -d "$site_location/test/" ]; then
    echo "ERROR: test subdirectory is required!"
    content=$(ls $site_location)
    echo "Content of $site_location is: $content" 
    usage
    exit -1
fi

echo "Input directory: $site_location is valid."

ds_name=$(basename "$site_location")
jn="autosplit-$ds_name"

# automatically fill in sbatch file
{
    echo "#!/bin/bash" 
    echo "#SBATCH -N 1" 
    echo "#SBATCH -n 1" 
    echo "#SBATCH -c 4" 
    echo "#SBATCH --mem 16G" 
    echo "#SBATCH -t 0-00:59"            
    echo "#SBATCH --job-name=\"$jn\""
    echo "#SBATCH --err=$base_shared_dir/log/job-%j.err" 
    echo "#SBATCH --output=$base_shared_dir/log/job-%j.out" 

    # actual job - ensure that the directories are correct - input and output!
    echo "singularity exec --pwd /home/ubuntu/image_dir --bind $code_repo/scripts:/home/ubuntu/ \
            --bind $code_repo/src/ohw:/home/ubuntu/ohw \
            --bind \"$site_location\":/home/ubuntu/image_dir \
            $base_shared_dir/pt-pyexiftool.simg python3 -u ../split_dataset.py \
            train"
} > "$jn.sh"

sbatch "$jn".sh
# cat "$jn".sh
rm "$jn".sh