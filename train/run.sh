#!/bin/bash

wget -O train.csv "https://raw.githubusercontent.com/jelambrar96-datatalks/house-price-predictor/refs/heads/main/dataset/train.csv"
# jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root # --IdentityProvider.token=$JUPYTER_TOKEN
python3 main.py

