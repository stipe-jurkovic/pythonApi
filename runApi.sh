#!/bin/bash

cd ~/pythonApi/
source ./apienv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000