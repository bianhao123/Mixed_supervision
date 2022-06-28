set -xv
CONFIG=$1
GPU=$2

nfold=(0 1 2 3)
nstage=("train" "test")
for fold in ${nfold[@]}
do
    for stage in ${nstage[@]}
    do
    python main.py --config $CONFIG --stage $stage --gpus $GPU --fold $fold
    done
done
python metric.py --config $CONFIG