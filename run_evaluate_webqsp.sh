#!/bin/bash

# GraphRAG CWQ 数据集评估脚本
# 使用 evaluate_test.py 进行测试集评估

python /root/autodl-tmp/SPARQ/evaluate_test.py \
    --test_graph_query_path /root/autodl-tmp/dataset/webqsp/graph_query/graph_query_test.json \
    --test_original_data_path /root/autodl-tmp/dataset/webqsp/webqsp/test_simple.json \
    --entities_path /root/autodl-tmp/dataset/webqsp/webqsp/entities.txt \
    --relations_path /root/autodl-tmp/dataset/webqsp/webqsp/relations.txt \
    --checkpoint_path /root/autodl-tmp/checkpoints/webqsp/best.pt \
    --bert_model_path /root/autodl-tmp/bert-base-uncased \
    --beam_width 7 \
    --max_path_length 3 \
    --top_k 10 \
    --output_path /root/autodl-tmp/output/test_results.json
