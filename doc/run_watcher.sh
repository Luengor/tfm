#!/bin/env bash

# Make a folder
mkdir -p /tmp/textidote

# Run the watcher in the background
bunx live-server --no-browser /tmp/textidote &

# Kill the watcher when the script is terminated
trap "pkill -f 'bunx live-server'" EXIT

# Run the main textidote loop
while true; do
    # Wait for changes
    inotifywait -e close_write informe.tex doc/*.tex

    # Run textidote
    textidote --check es --output html --read-all informe.tex > /tmp/textidote/output.html.tmp

    # Move
    mv /tmp/textidote/output.html.tmp /tmp/textidote/output.html

    # Wait a bit before the next check
    sleep 1
done

