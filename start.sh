#!/bin/bash

sleep 15
cd /opt/dorylus
source .venv/bin/activate
export CHAT_OPENAI_KEY=''
python ./main.py
