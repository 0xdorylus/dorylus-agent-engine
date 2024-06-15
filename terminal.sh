#!/bin/bash

while true
do
    DATA=$(netstat -anp | grep LISTEN | grep ':5001')
    if [ -n "$DATA" ]; then
        break
    else
        echo "wait and check again ..."
        sleep 3
    fi
done

echo "wait for connected speech done ..."
sleep 15

cd /opt/dorylus
source .venv/bin/activate
python ./terminal.py
