#!/bin/zsh

PYTHON=$(which python)

values=("1" "2" "3" "4" "5")
for RUN_NAME in "${values[@]}"; do
    echo "Running train data"
    export $(grep -v '^#' ../.env | gxargs -d '\n') && $PYTHON ./populate-train-pre-computed.py -f $RUN_NAME

    echo "Running test data"
    export $(grep -v '^#' ../.env | gxargs -d '\n') && $PYTHON ./test-pre-computed-hits-only.py -f $RUN_NAME
done