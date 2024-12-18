#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1 
#SBATCH -c 1 
#SBATCH --mem 500M 
#SBATCH --time=0-07:00
#SBATCH --job-name=ctrl_{{jid0}}
#SBATCH --err=/mnt/scratch_lustre/hawkweed_drone_scratch/log/ctrl_%j.err
#SBATCH --output=/mnt/scratch_lustre/hawkweed_drone_scratch/log/ctrl_%j.out


# Job dependencies
jid1={{jid0}}
jid1={{jid1}}
jid2={{jid2}}
jid3={{jid3}}

echo "Monitoring Job IDs: $jid0, $jid1, $jid2, $jid3"

# Function to get job status
get_job_status() {
    local job_id=$1
    status=$(sacct -j "$job_id" --format=State --noheader | awk '{print $1}')
    echo "$status"
}

# Check and process each job sequentially
process_jobs() {
    local current_job=$1
    shift
    local remaining_jobs=("$@")

    while true; do
        status=$(get_job_status "$current_job")
        echo "Job $current_job status: $status"

        if [[ "$status" == "COMPLETED" ]]; then
            echo "Job $current_job completed successfully."
            echo "Cancelling remaining jobs: ${remaining_jobs[*]}"
            for job in "${remaining_jobs[@]}"; do
                echo "Cancelling job $job"
                scancel "$job"
            done
            return 0
        elif [[ "$status" == "FAILED" || "$status" == "CANCELLED" ]]; then
            echo "Job $current_job failed or was cancelled. Proceeding to next job..."
            break
        fi
        sleep 60
    done

    if [[ ${#remaining_jobs[@]} -gt 0 ]]; then
        process_jobs "${remaining_jobs[0]}" "${remaining_jobs[@]:1}"
    else
        echo "No more jobs to process."
    fi
}

# Start processing jobs
process_jobs "$jid0" "$jid1" "$jid2" "$jid3"

echo "Controller script finished."
