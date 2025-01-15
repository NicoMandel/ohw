#!/bin/bash

# code and data reposiories
code_repo="/mnt/appsource/local/hawkweed_drone/ohw"
base_shared_dir="/mnt/scratch_lustre/hawkweed_drone_scratch"

usage() {
    printf "\nUsage : $0 -i <input>
    Options:
        -i input folder. Inside, requires a subfolder named <images> and one named <labels>. Will create <crops> subfolder automatically, with <images> and adapted <labels> subfolder.
        CAREFUL. DO NOT USE DIRECTLY DETECTION OUTPUT. The visualisations contain bounding boxes, which will be cropped too, so that will taint the image. Copy across the original images first.
        -h display this help message
        
        Example:
            $0 -s '/path/to/folder' 
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
if [  ! -d "$site_location/images/" ]; then
    echo "ERROR: images subdirectory is required!"
    content=$(ls $site_location)
    echo "Content of $site_location is: $content"
    usage
    exit -1
fi

if [  ! -d "$site_location/labels/" ]; then
    echo "ERROR: label subdirectory is required!"
    content=$(ls $site_location)
    echo "Content of $site_location is: $content" 
    usage
    exit -1
fi

echo "Input directory: $site_location is valid. Creating crop subdirectory"

# creating subdirectory.
mkdir -p "$site_location/crops/images"
mkdir -p "$site_location/crops/labels"

ds_name=$(basename "$site_location")
jn="crops-$ds_name"

# Creating log directory
mkdir -p "$base_shared_dir/log/crops/$ds_name"

# automatically fill in sbatch file
{
    echo "#!/bin/bash" 
    echo "#SBATCH -N 1" 
    echo "#SBATCH -n 1" 
    echo "#SBATCH -c 4" 
    echo "#SBATCH --mem 32G" 
    echo "#SBATCH -t 0-02:59"            
    echo "#SBATCH --job-name=\"$jn\""
    echo "#SBATCH --err=$base_shared_dir/log/crops/$ds_name/job-%j.err" 
    echo "#SBATCH --output=$base_shared_dir/log/crops/$ds_name/job-%j.out" 

    # actual job - ensure that the directories are correct - input and output!
    echo "singularity exec --pwd /home/ubuntu --bind $code_repo/scripts:/home/ubuntu/ \
            --bind $code_repo/src/ohw:/home/ubuntu/ohw \
            --bind \"$site_location\":/home/ubuntu/image_dir \
            $base_shared_dir/pt-pyexiftool.simg python3 -u crop_images.py \
            image_dir/images/ image_dir/labels/ -o $site_location/crops"
} > "$jn.sh"

sbatch "$jn".sh
# cat "$jn".sh
rm "$jn".sh