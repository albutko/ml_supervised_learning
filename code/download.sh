#!/bin/bash

data_dir='../data'
higgs_dir=$data_dir'/higgs'
poker_dir=$data_dir'/poker'

base_url="https://archive.ics.uci.edu/ml/machine-learning-databases"
higgs_data=$base_url'/00280/HIGGS.csv.gz'
poker_train=$base_url'/poker/poker-hand-training-true.data'
poker_test=$base_url'/poker/poker-hand-testing.data'

if [ -d $data_dir ]; then
    echo 'Data already downloaded'
else
    mkdir $data_dir
    mkdir $higgs_dir
    mkdir $poker_dir
fi

wget -P $poker_dir -i $poker_train -O poker_train.csv --progress=bar
