#!/usr/bin/env bash

RUNID=$1
DRY_RUN=$2

if [ $DRY_RUN -eq 0 ]
then
    echo_and_run() { echo "$@" ; "$@" ; }
else
    echo_and_run() { echo "$@" ; }
fi

ENV="Breakout-v3"
NET="--hidden-size 32  --num-hidden 3 --lr 0.001 --use-conv-nets"
EXPLORE="--eps-min 0.01 --eps-decay 0.95 --eps-update-steps 100"
TRAIN_EVAL="--max-steps 40000 --eval-steps 20000 --train-epoch-steps 20000 --eval-epoch-steps 5000"

COMMON_PARAMS="--env ${ENV} --target --gamma 0.99 --log-dir logs_dev --target-update-steps 100"
RUN_NAME="--run-name run-$RUNID"

echo_and_run python dqn.py $RUN_NAME $COMMON_PARAMS $TRAIN_EVAL $NET $EXPLORE
