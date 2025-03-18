#!/bin/zsh

RUN_NAME="1"
PYTHON=$(which python)

echo "Running train data"
export $(grep -v '^#' ../.env | gxargs -d '\n') && $PYTHON ./populate-train.py -f $RUN_NAME

echo "Running test data"
export $(grep -v '^#' ../.env | gxargs -d '\n') && $PYTHON ./test.py -f $RUN_NAME