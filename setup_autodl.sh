#!/bin/bash

# Setup Script for AutoDL / Linux Isolation
# This script creates a virtual environment and initializes the project.

echo -e "\e[36m--- Initializing AutoDL RAG Environment ---\e[0m"

# 1. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 2. Upgrade pip and install requirements
echo "Installing dependencies..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

# 3. Create necessary folders
folders=("models" "data" "vector_db")
for f in "${folders[@]}"; do
    if [ ! -d "$f" ]; then
        mkdir -p "$f"
        echo "Created folder: $f"
    fi
done

echo -e "\n\e[32mSetup Complete!\e[0m"
echo -e "\e[33mTo run the app, use: ./.venv/bin/python app.py\e[0m"
