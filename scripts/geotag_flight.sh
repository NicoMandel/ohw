#!/bin/bash

# code and data reposiories
code_repo="/mnt/appsource/local/hawkweed_drone/ohw"
base_shared_dir="/mnt/scratch_lustre/hawkweed_drone_scratch"

usage() {
    printf "\nUsage : $0 -s <directory> -c <csv_file> -k -n
    Options:
        -s image directory which to process. Images have to be contained directly in this. Cannot contain child directories, such as <images> or others.
        -c csv file to use for geotagging. Will load and look for correspondence of image_id beween images in the folder and in this csv.
        -k whether to create kml files or not.
        -n whether to create new images with north arrows or not

        -h display this help message
        
        Example:
            $0 -s '/path/to/site/detections/flight' -c '/path/to/geotag.csv' -k -n
"
}

# default values
site_location=""
csv_file=""
kml_flag=""
north_flag=""

# argument parsing
while getopts 's:c:knh' flag; do 
    case "${flag}" in
        s) site_location="${OPTARG}" ;;
        c) csv_file="${OPTARG}" ;;
        k) kml_flag="--kml" ;;
        n) north_flag="--north" ;;
        h) usage 
            exit -1;;
        *) usage
            exit -1;;
    esac
done

# check required arguments
if [ -z "$site_location" ]; then
    echo "ERROR: Image directory is required!"
    usage
    exit -1
fi

if [ -z "$csv_file" ]; then
    echo "ERROR: csv file is required!"
    usage
    exit -1
fi

echo "Flight: $site_location"
echo "CSV File: $csv_file"

# getting the csv_file parameters
csv_base=$(dirname "$csv_file")
csv_name=$(basename "$csv_file")

module load slurm
# echo all the relevant factors into the base file by structure
jn="Geotag-$csv_name"

# automatically fill in sbatch file
{
    echo "#!/bin/bash" 
    echo "#SBATCH -N 1" 
    echo "#SBATCH -n 1" 
    echo "#SBATCH -c 4" 
    echo "#SBATCH --mem 32G" 
    echo "#SBATCH -t 0-02:59"            
    echo "#SBATCH --job-name=\"$jn\"" 
    echo "#SBATCH --err=$base_shared_dir/log/geotag/job-%j.err" 
    echo "#SBATCH --output=$base_shared_dir/log/geotag/job-%j.out" 

    # actual job - ensure that the directories are correct - input and output!
    echo "singularity exec --nv --pwd /home/ubuntu --bind $code_repo/scripts:/home/ubuntu/ \
            --bind $code_repo/src/ohw:/home/ubuntu/ohw \
            --bind \"$site_location\":/home/ubuntu/geotags \
            --bind \"$csv_base\":/home/ubuntu/csv \
            $base_shared_dir/container_hist/yolo-kml.simg python3 -u geotag_images.py \
            geotags csv/$csv_name $kml_flag $north_flag" 
} > "$jn.sh"

sbatch "$jn".sh
# cat "$jn".sh
rm "$jn".sh