#!/bin/bash

# GraphRAG CWQ 数据集评估脚本
# 使用 evaluate_test.py 进行测试集评估

python evaluate_test.py \
    --test_graph_query_path /root/autodl-tmp/dataset/CWQ/graph_query/graph_query_test.json \
    --test_original_data_path /root/autodl-tmp/dataset/CWQ/CWQ/test_simple.json \
    --entities_path /root/autodl-tmp/dataset/CWQ/CWQ/entities.txt \
    --relations_path /root/autodl-tmp/dataset/CWQ/CWQ/relations.txt \
    --checkpoint_path /root/autodl-tmp/checkpoints/best.pt \
    --bert_model_path /root/autodl-tmp/bert-base-uncased \
    --beam_width 3 \
    --max_path_length 3 \
    --top_k 10 \
    --output_path /root/autodl-tmp/output/test_results.json
