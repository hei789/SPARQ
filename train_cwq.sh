#!/bin/bash

# GraphRAG CWQ 数据集快速训练脚本
# 使用 train_fast.py 进行训练

python SPARQ/train_fast.py \
    --train_graph_query_path /root/autodl-tmp/dataset/CWQ/graph_query/graph_query_train.json \
    --train_original_data_path /root/autodl-tmp/dataset/CWQ/CWQ/train_simple.json \
    --dev_graph_query_path /root/autodl-tmp/dataset/CWQ/graph_query/graph_query_dev.json \
    --dev_original_data_path /root/autodl-tmp/dataset/CWQ/CWQ/dev_simple.json \
    --entities_path /root/autodl-tmp/dataset/CWQ/CWQ/entities.txt \
    --relations_path /root/autodl-tmp/dataset/CWQ/CWQ/relations.txt \
    --bert_model_path /root/autodl-tmp/bert-base-uncased \
    --epochs 10 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_negatives 2 \
    --use_amp \
    --cache_paths
