# !/bin/bash

# Find workspace folder and add it as safe directory
git config --global --add safe.directory "/workspaces/Code"

# Installing search and robot locally
pip3 install -e .