#!/bin/bash

DATASET="qm9"
DATASETPATH="./data/qm9/dsgdb9nsd/"
MODEL="MPNNv2" # MPNNv1 uses too much memory, and MPNNv3 crashed
EPOCHS="1" 
PROFILE_EPOCHS="0"
PROFILE_BATCHES="40-42,500-502,700-702"
PROF_ROOT="./profile"
LOG_ROOT="./log"

DATE=`date '+%Y_%m_%d__%H_%M_%S'`
LOG_DIR="$LOG_ROOT/$DATE"
CONSOLE_LOG="$LOG_ROOT/$DATE.out"
PROF_DIR="$PROF_ROOT/$DATE/"


COMMAND="python main.py \
  --dataset $DATASET \
  --datasetPath $DATASETPATH \
  --epochs $EPOCHS \
  --model $MODEL \
  --log-interval 20 \
  --profile-epoch-list $PROFILE_EPOCHS \
  --profile-batch-list $PROFILE_BATCHES \
  --logPath $LOG_DIR/ \
  --consoleLogPath $CONSOLE_LOG"

$COMMAND 

