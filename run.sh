#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

OUTPUT_FOLDER='/proj/karst/results/siamese/'$DTime'_siamese'

python src/main.py \
--my-config "config.conf" \
--save_dir $OUTPUT_FOLDER