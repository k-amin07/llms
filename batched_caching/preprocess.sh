#!/bin/zsh

PYTHON=$(which python)

# echo "Creating embeddings"
# export $(grep -v '^#' ../.env | gxargs -d '\n') && $PYTHON ./create_embeddings.py
# echo "Creating train/test splits"
# export $(grep -v '^#' ../.env | gxargs -d '\n') && $PYTHON ./create-train-test-balanced-complete-toxic.py
# echo "Train/test splits created

values=("1")
#  "2" "3" "4" "5")
for RUN_NAME in "${values[@]}"; do
    echo "Running test data"
    export $(grep -v '^#' ../.env | gxargs -d '\n') && caffeinate -i $PYTHON ./add_llm_ratings_to_data.py -f $RUN_NAME
done