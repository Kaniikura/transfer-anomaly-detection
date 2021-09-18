#!/bin/zsh

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -m abe

# environment
source /home/acc12901zs/.zshrc

source /etc/profile.d/modules.sh
module load cuda/10.2
module load cudnn/8.0/8.0.5
module load nccl/2.8/2.8.3-1
module load gcc/7.4.0
module load python/3.8/3.8.2

conda activate rapids

# clone git repo & cd
REPO=transfer-anomaly-detection
git clone https://Kaniikura:${MY_GIT_TOKEN}@github.com/Kaniikura/${REPO} $SGE_LOCALDIR/${REPO}
rm -rf ${SGE_LOCALDIR}/${REPO}/data
ln -s /home/${ID_USER}/Git/${REPO}/data/ ${SGE_LOCALDIR}/${REPO}/data
ln -s /home/${ID_USER}/Git/${REPO}/cache/ ${SGE_LOCALDIR}/${REPO}/cache
cd $SGE_LOCALDIR/${REPO}

# parse args
for opt in "$@"; do
    case $opt in
        -p|--yaml_path) YAML_PATHS+=($2);;
        -s|--sweep_id) SWEEP_IDS+=($2);;
        -*)
            echo "$PROGNAME: illegal option -- '$(echo $1 | sed 's/^-*//')'" 1>&2
            exit 1;;
    esac
    shift
done

# create new sweep
for yaml_path in "${YAML_PATHS[@]}"; do
    sweep_output=$(wandb sweep $yaml_path 2>&1)
    sid=`echo ${sweep_output} | sed -u -ne 's/^wandb: Run sweep agent with: wandb agent //p'`
    SWEEP_IDS+=($sid)
done

# start new sweeps or resume from existing ones
for sid in "${SWEEP_IDS[@]}"; do
    wandb agent ${sid} >/dev/null 2>&1 &
    sleep 60
done

# exit when all sweeps are completed
counter=0
patience=2
while true;
do
    # find agent processes
    num_agents=`ps aux | grep "wandb agent" | grep python | wc -l`
    if [[ "$num_agents" -eq "0" ]]; then
        let counter++
        if [[ "$counter" -gt "$patience" ]]; then
            exit
        fi
    else
        counter=0
    fi
    sleep 60
done
