#!/bin/bash

python transformers/run_sfda_negation.py --model_name_or_path tmills/roberta_sfda_sharpseed --data_dir $1 --output_dir $2 --cache ./cache --do_predict

