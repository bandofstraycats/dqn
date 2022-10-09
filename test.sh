#!/usr/bin/env bash

RUNID=$1
DRY_RUN=$2

if [ $DRY_RUN -eq 0 ]
then
    echo_and_run() { echo "$@" ; "$@" ; }
else
    echo_and_run() { echo "$@" ; }
fi

ENV="LunarLander-v2"
NET="--hidden-size 64 --num-hidden 2"
EXPLORE="--eps-min 0.01 --eps-decay 0.995 --eps-update-steps 1000"
TRAIN_EVAL="--train-steps 500000 --test-steps 10000 --nb-epoch 1"
LOG="--log-dir logs_dev --log-steps 10000"

COMMON_PARAMS="--env ${ENV} --target --gamma 0.99 --target-update-steps 100"
RUN_NAME="--run-name run-$RUNID"

echo_and_run python dqn.py $RUN_NAME $COMMON_PARAMS $TRAIN_EVAL $LOG $NET $EXPLORE
