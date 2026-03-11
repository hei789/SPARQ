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
      --entity_names_path ./dataset/CWQ/CWQ/entity_name.txt \
      --bert_model_path /root/autodl-tmp/bert-base-uncased \
      --checkpoint_dir "/root/autodl-tmp/checkpoints/CWQ" \
      --path_encoder_type gnn \
      --path_encoder_layers 2 \
      --epochs 30 \
      --batch_size 8 \
      --gradient_accumulation_steps 8 \
      --num_negatives 2 \
      --use_amp \
      --cache_paths \
      --num_relations 100
