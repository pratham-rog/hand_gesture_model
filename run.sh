#!/bin/zsh
# Run the virtual keyboard — sets up venv if needed
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
fi

venv/bin/python key.py
