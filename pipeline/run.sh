#!/bin/zsh

PYTHON=$(which python)

set -o allexport
source ../.env
set +o allexport

if [ ! -f ./toxicity_ratings_embeddings.json ]; then
    echo "Creating Embeddings"
    $PYTHON ./1-create-embeddings.py
fi

mkdir -p ./batches

if [ ! -f ./batches/batch_input_1.jsonl ]; then
    echo "Creating OpenAI batches"
    $PYTHON ./2-create_openai_batches.py
fi

if [ ! -f ./batches/batch_output_1.jsonl ]; then
    echo "Processing OpenAI batches"
    $PYTHON ./3-process_openai_batches.py
fi

if [ ! -f ./toxicity_ratings_embeddings_llms.json ]; then
    echo "Processing LLMs"
    $PYTHON ./4-data_process_llm_ratings.py
    $PYTHON ./5-data_add_openai_ratings.py
fi

echo "Creating Train Test Splits"

$PYTHON ./6-data_train_test_split.py


values=("1" "2" "3" "4" "5")

for RUN_NAME in "${values[@]}"; do
    echo "Running train data"
    $PYTHON ./7-populate_train_data.py -f $RUN_NAME
    echo "Running test data"
    $PYTHON ./8-process_test_data.py -f $RUN_NAME
    echo "Computing metrics"
    $PYTHON ./9-compute_metrics.py -f $RUN_NAME
done