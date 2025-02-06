#!/bin/zsh

# Help menu function
show_help() {
    help_menu="""Usage: run.sh [FILE_PATH]

Run a Python script from the examples_* directories.

Options:
    FILE_PATH    (Optional) Path to the specific file to run
    --help       Show this help menu
"""
    echo $help_menu
}

# Check for --help argument
if [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

if [ ! -f ./.env ]; then
    echo "No .env file found. Generating one. Please add the relevant API keys to it"
    echo "OPENAI_API_KEY=\nTAVILY_API_KEY=\n" >> .env
    exit 1
fi

PS3="Please choose a file to run: "
FILES=$(ls "examples"**/*/*)
PYTHON=$(which python)

if test -z "$1"
then
    select FILE_PATH in Quit "${=FILES}"
    do
        if [ "$FILE_PATH" = "Quit" ]; then
            break
        fi
        echo ""
        echo "Running $FILE_PATH"
        echo ""
        export $(grep -v '^#' .env | gxargs -d '\n') && $PYTHON ./$FILE_PATH
        echo ""
    done
else
    FILE_PATH=$1
    export $(grep -v '^#' .env | gxargs -d '\n') && $PYTHON ./$FILE_PATH
fi
