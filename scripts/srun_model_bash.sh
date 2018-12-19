queue=${1:-dev}
hour=${2:-24}

srun --job-name=${jname} \
    --gres=gpu:8 -c 48 -C volta -v \
    --partition=${queue} --comment "Deadline for NAACL 12/10" \
    --time=${hour}:00:00 --mem 128GB --pty \
    bash
