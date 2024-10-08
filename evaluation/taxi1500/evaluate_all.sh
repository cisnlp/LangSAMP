#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#MODEL=${1:-cis-lmu/glot500-base}
MODEL="xlm-roberta-base"
GPU=${2:-2}

export CUDA_VISIBLE_DEVICES=$GPU
MODEL_TYPE="xlmr"

OUTPUT_DIR="/mounts/data/proj/yihong/decoupled_training/evaluation/taxi1500/results/"
init_checkpoint="/mounts/data/proj/yihong/decoupled_training/baseline_model"
eng_data_dir="/mounts/data/proj/chunlan/Taxi1500_data/Glot500_data/eng_data"
test_data_dir="/mounts/data/proj/chunlan/Taxi1500_data/Glot500_data/lrs_test2"

python -u evaluate_all.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --epochs 40 \
    --init_checkpoint $init_checkpoint \
    --eng_data_dir $eng_data_dir \
    --test_data_dir $test_data_dir \
    --nr_of_seeds 4

