#!/bin/bash

# Start Backend API Server
echo "Starting Opinion-Based Search Backend API..."
cd "$(dirname "$0")/backend"
python app.py

