script=${1}
queue=${2:-dev}
mode=${3:-train}
jname=${4:-TransformerB}
load_from=${5:-}
hour=${6:-24}

srun --job-name=${jname} \
    --gres=gpu:8 -c 48 -C volta -v \
    --partition=${queue} --comment "Deadline for NAACL 12/10" \
    --time=${hour}:00:00 --mem 128GB --pty \
    bash ${script} 8 ${jname} ${mode} ${load_from}
