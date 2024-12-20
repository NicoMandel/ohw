#!/bin/bash

# code and data reposiories
code_repo="/mnt/appsource/local/hawkweed_drone/ohw"
base_shared_dir="/mnt/scratch_lustre/hawkweed_drone_scratch"
resolution="024cm" # alternative -> "1cm"
model_registry="results/model_res.xlsx"

usage() {
    printf "\nUsage : $0 -d <directory>
    Options:
        -s directory which to process. Must contain child directories containing <flight> (case insensitive),
            which have to contain the images directly. Cannot contain child directories, such as <images> or others.
            The <flight> diretories will be processed, one job submitted for each and outputs written to:
            <$base_shared_dir/results_nico>
            Example directories are:
                '/mnt/load/aerial/drone_uav/2023-24/NPWS_OHW_4/23.12.18 - Yarrabee North'
                /mnt/load/aerial/drone_uav/2023-24/Ag_Drones_1/2023.12.19_Tantangara_North_East/WingtraPilotProjects/ 
                alternatives: , 2023.12.22_Mufflers_Gap, 202.12.23_Long_Plain_Rd_Snowy_Mtns_Hwy_jxn, 2023.12.21_Billmans_Point/
        -r resolution which to process. Must be one of <024cm> or <1cm>. So that the appropriate model can be chosen. Defaults to 024cm
        -m path to model registry file, specifying models that can be chosen. Defaults to '$base_shared_dir/$model_registry' 
        -h display this help message
        
        Example:
            $0 -s '/path/to/site' -r '024cm'
"
}

# default values
resolution="024cm"
site_location=""

# argument parsing
while getopts 's:r:m:h' flag; do 
    case "${flag}" in
        s) site_location="${OPTARG}" ;;
        r) resolution="${OPTARG}" ;;
        m) model_registry="${OPTARG}" ;;
        h) usage 
            exit -1;;
        *) usage
            exit -1;;
    esac
done

# check required arguments
if [ -z "$site_location" ]; then
    echo "ERROR: Site location is required!".
    usage
    exit -1
fi

echo "Site Location: $site_location"
echo "Resolution: $resolution"
echo "Model registry file: $model_registry"

# automated job naming and finding subdirectories
sitename=$(basename "$site_location")
mapfile -t flightdirs < <(find "$site_location" -maxdepth 1 -type d -iname "*flight*")

# inside the site location, create a directory
mkdir -p "$site_location/detections"
echo "created $site_location/detections"

module load slurm
# echo all the relevant factors into the base file by structure
for jobsite in "${flightdirs[@]}"; do
    # create job name
    jobname=$(basename "$jobsite")
    jn="$sitename-$jobname"

    # create output directory for the job as subdir of "detections"
    j_output="$site_location/detections/$jobname"
    mkdir -p $j_output
    echo "created $j_output"

    # automatically fill in sbatch file
    {
        echo "#!/bin/bash" 
        echo "#SBATCH -N 1" 
        echo "#SBATCH -n 1" 
        echo "#SBATCH -c 4" 
        echo "#SBATCH --mem 32G" 
        echo "#SBATCH --partition=GPU" 
        echo "#SBATCH --gpus-per-node=1" 
        echo "#SBATCH -t 0-02:59"            
        echo "#SBATCH --job-name=\"$jn\"" 
        echo "#SBATCH --err=$base_shared_dir/log/inference/job-%j.err" 
        echo "#SBATCH --output=$base_shared_dir/log/inference/job-%j.out" 

        # actual job - ensure that the directories are correct - input and output!
        echo "singularity exec --nv --pwd /home/ubuntu --bind $code_repo/scripts:/home/ubuntu/ \
                --bind $code_repo/src/ohw:/home/ubuntu/ohw \
                --bind \"$jobsite\":/home/ubuntu/inference \
                --bind $j_output:/home/ubuntu/inference_out \
                --bind $base_shared_dir/results:/home/ubuntu/results \
                $base_shared_dir/pt-sahi-123.simg python3 -u inference_sahi.py \
                inference $model_registry $resolution inference_out -n \"$sitename/$jobname\" -s -v" 
    } > "$jn.sh"

    # choose between sbatch "$jn.sh" or cat "$jn.sh"
    jid0=$(sbatch "$jn.sh" | awk '{print $NF}')

    # resubmit twice - for breaking case
    echo "$jid0"
    jid1=$(sbatch --dependency=afternotok:$jid0 "$jn.sh" | awk '{print $NF}')
    jid2=$(sbatch --dependency=afternotok:$jid1 "$jn.sh" | awk '{print $NF}')
    jid3=$(sbatch --dependency=afternotok:$jid2 "$jn.sh") 

    # cat "$jn".sh
    rm "$jn".sh
done