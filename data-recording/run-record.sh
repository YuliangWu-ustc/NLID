#!/bin/bash
set -e

# check the number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <folderPath> <beginIndex> <endIndex>"
    exit 1
fi

# get the folderPath, beginIndex and endIndex from command line arguments
folderPath=$1
beginIndex=$2
endIndex=$3

logFile="$(date +%Y-%m-%d_%H-%M-%S).log"
absoluteLogFile="$folderPath/$logFile"

# check if the folder exists, if not, create it
if [ ! -d "$folderPath" ]; then
    mkdir -p "$folderPath"
fi

echo "absoluteLogFile: $absoluteLogFile"

# check if the log file already exists
if [ ! -f "$absoluteLogFile" ]; then
    # if the log file does not exist, run the command, ignoring DEBUG in the output
    python3 run-record.py "$folderPath" "$beginIndex" "$endIndex" 2>&1 | tee "$absoluteLogFile" | grep -v DEBUG
fi
