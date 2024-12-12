#!/bin/bash

# code and data reposiories
code_repo="/home/mandeln/ohw"
base_shared_dir="/mnt/scratch_lustre/hawkweed_drone_scratch"
resolution="024cm" # alternative -> "1cm"
model_registry="results/model_res.xlsx"

usage() {
    printf "\nUsage : $0 -d <directory>
    Options:
        -d directory which to process. Must contain child directories containing <flight> (case insensitive),
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

module load slurm
# echo all the relevant factors into the base file by structure
for jobsite in "${flightdirs[@]}"; do
    # create job name
    jobname=$(basename "$jobsite")
    jn="$sitename-$jobname"

    # automatically fill in sbatch file
    {
        echo "#!/bin/bash" 
        echo "#SBATCH -N 1" 
        echo "#SBATCH -n 1" 
        echo "#SBATCH -c 4" 
        echo "#SBATCH --mem 32G" 
        echo "#SBATCH --partition=GPU" 
        echo "#SBATCH --gpus-per-node=1" 
        echo "#SBATCH --mem 32G" 
        echo "#SBATCH -t 0-00:59"            
        echo "#SBATCH --job-name=\"$jn\"" 
        echo "#SBATCH --err=$base_shared_dir/log_nico/inference/job-%j.err" 
        echo "#SBATCH --output=$base_shared_dir/log_nico/inference/job-%j.out" 

        # module parts
        echo "module purge" 
        echo "module load go singularity"        

        # actual job - ensure that the directories are correct - input and output!
        echo "singularity exec --nv --pwd /home/ubuntu --bind $code_repo/scripts:/home/ubuntu/ \
                --bind $code_repo/src/ohw:/home/ubuntu/ohw \
                --bind \"$jobsite\":/home/ubuntu/inference \
                --bind $base_shared_dir/inference_out:/home/ubuntu/inference_out \
                --bind $base_shared_dir/results_nico:/home/ubuntu/results \
                $base_shared_dir/pt-sahi-123.simg python3 -u inference_sahi.py \
                inference $model_registry $resolution inference_out -n \"$sitename/$jobname\" -s -v --debug" 
    } > "$jn.sh"

    # choose between sbatch "$jn.sh" or cat "$jn.sh"
    jid0=$(sbatch "$jn.sh" | awk '{print $NF}')

    # resubmit twice - for breaking case
    echo "$jid0"
    jid1=$(sbatch --dependency=afternotok:$jid0 "$jn.sh" | awk '{print $NF}')
    jid2=$(sbatch --dependency=afternotok:$jid1 "$jn.sh")

    # cat "$jn".sh
    rm "$jn".sh
done