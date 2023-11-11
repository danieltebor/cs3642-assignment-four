#!/bin/sh
cd "$(dirname "$0")"

echo "Installing python packages"
pip install --user -r requirements.txt

echo "Configuring project"
cmake -B build -S .