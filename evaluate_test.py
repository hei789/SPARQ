#!/usr/bin/env python3
"""
GraphRAG 测试集评估脚本
计算 Hit@1 ~ Hit@10 指标
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_query_encoder.models import GraphQueryEncoder
from graph_retriever import GraphRetriever
from main import CWQDataset, GraphRAGConfig, SubgraphProcessor


# 兼容 train_fast.py 保存的检查点
@dataclass
class FastTrainingConfig:
    """占位类，用于加载 train_fast.py 保存的检查点"""
    pass


@dataclass
class EvalConfig:
    """评估配置"""
    # 数据路径
    test_graph_query_path: str
    test_original_data_path: str
    entities_path: str
    relations_path: str

    # 模型配置
    checkpoint_path: str
    bert_model_path: str = "bert-base-uncased"
    hidden_dim: int = 768
    num_query_layers: int = 3
    num_retriever_layers: int = 2
    num_relations: int = 100

    # 推理配置
    beam_width: int = 3
    max_path_length: int = 3
    similarity_threshold: float = 0.0
    top_k: int = 10  # 计算Hit@K的最大K值

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples: Optional[int] = None
    output_path: Optional[str] = None


class GraphRAGEvaluator:
    """GraphRAG评估器"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._load_model()
        self.dataset = None
        self.subgraph_processor = None

    def _load_model(self):
        """加载训练好的模型"""
        print(f"Loading checkpoint from {self.config.checkpoint_path}")

        # 创建模型
        query_encoder = GraphQueryEncoder(
            hidden_dim=self.config.hidden_dim,
            num_gnn_layers=self.config.num_query_layers,
            num_relations=self.config.num_relations,
            bert_model_name=self.config.bert_model_path,
            use_query_centered_pooling=True,
            use_bfs_gnn=True
        )

        retriever = GraphRetriever(
            hidden_dim=self.config.hidden_dim,
            num_gnn_layers=self.config.num_retriever_layers,
            num_relations=self.config.num_relations,
            beam_width=self.config.beam_width,
            max_path_length=self.config.max_path_length,
            similarity_threshold=self.config.similarity_threshold
        )

        # 修复：保存完整的 retriever，不只是 path_encoder
        model = nn.ModuleDict({
            'query_encoder': query_encoder,
            'retriever': retriever
        }).to(self.device)

        # 加载检查点
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

            # 检查是否是旧版 checkpoint（只有 path_encoder）
            if 'path_encoder' in state_dict and 'retriever' not in state_dict:
                print("Loading legacy checkpoint (path_encoder only)...")
                # 将旧版 path_encoder 权重迁移到新结构
                retriever.path_encoder.load_state_dict(state_dict['path_encoder'])
                # query_encoder 直接加载
                if 'query_encoder' in state_dict:
                    model['query_encoder'].load_state_dict(state_dict['query_encoder'])
                print("Warning: GNN layers in retriever are randomly initialized!")
                print("Please retrain the model to get the full retriever weights.")
            else:
                # 正常加载完整 checkpoint
                model.load_state_dict(state_dict)

            print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"Checkpoint metrics: {checkpoint.get('metrics', 'unknown')}")
        else:
            # 直接加载 state dict（兼容旧格式）
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")

        model.eval()
        print("Model loaded successfully")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return model

    def set_dataset(self, dataset: CWQDataset):
        """设置数据集"""
        self.dataset = dataset
        self.subgraph_processor = SubgraphProcessor(
            entities=dataset.entities,
            relations=dataset.relations,
            hidden_dim=self.config.hidden_dim,
            device=self.device,
            num_relations=self.config.num_relations
        )

    def extract_answer_entities(self, sample: Dict) -> Set[int]:
        """
        从样本中提取答案实体索引

        返回:
            answer_entity_indices: 答案实体的索引集合（在entities.txt中的索引）
        """
        answer_entities = set()

        # 从answers字段提取
        answers = sample.get('answers', [])
        if answers:
            for ans in answers:
                if isinstance(ans, dict):
                    # 如果answers包含实体信息
                    if 'kb_id' in ans:
                        # 尝试找到对应的实体索引
                        kb_id = ans['kb_id']
                        if kb_id in self.dataset.entities:
                            idx = self.dataset.entities.index(kb_id)
                            answer_entities.add(idx)
                        elif kb_id.startswith('Q'):
                            # 尝试匹配不同格式
                            for i, entity in enumerate(self.dataset.entities):
                                if kb_id in entity or entity in kb_id:
                                    answer_entities.add(i)
                                    break
                elif isinstance(ans, int):
                    answer_entities.add(ans)

        # 如果没有找到答案实体，尝试从其他字段提取
        if not answer_entities:
            # 尝试从subgraph的tuples中提取可能的答案
            subgraph = sample.get('subgraph', {})
            tuples = subgraph.get('tuples', [])
            # 收集所有作为object出现的实体
            for h, r, t in tuples:
                answer_entities.add(t)

        return answer_entities

    def evaluate_sample(self, idx: int) -> Dict[str, Any]:
        """
        评估单个样本

        返回:
            result: 包含预测路径和命中情况的字典
        """
        sample = self.dataset.get_sample(idx)
        if sample is None:
            return {"error": "Sample not found"}

        question = sample['question']
        triples = sample['triples']
        subgraph_data = sample['subgraph']
        topic_entities = sample['entities']

        if not subgraph_data or 'tuples' not in subgraph_data:
            return {"id": sample['id'], "error": "No subgraph data"}

        # 提取答案实体
        answer_entities = self.extract_answer_entities(sample)

        # 编码查询
        with torch.no_grad():
            query_result = self.model['query_encoder'](triples, return_all_node_features=True)
            query_embedding = query_result[1] if isinstance(query_result, tuple) else query_result

        # 构建子图
        node_features, edge_index, edge_types, entity_idx2local = \
            self.subgraph_processor.build_subgraph(subgraph_data, topic_entities)

        if node_features is None or node_features.size(0) == 0:
            return {"id": sample['id'], "error": "Empty subgraph"}

        # 创建反向映射：局部索引 -> 实体行号
        local2entity_idx = {v: k for k, v in entity_idx2local.items()}

        # 构建关系映射：局部关系索引 -> 关系名称
        # 注意：edge_types 中的值是 relation2idx[r] = next_rel_idx % num_relations
        # 所以需要建立反向映射
        relation_line2local = {}  # 关系行号 -> 局部索引
        local2relation_line = {}  # 局部索引 -> 关系行号（取第一个）
        next_rel_idx = 0
        for h, r, t in subgraph_data.get('tuples', []):
            if r not in relation_line2local:
                local_idx = next_rel_idx % self.config.num_relations
                relation_line2local[r] = local_idx
                if local_idx not in local2relation_line:
                    local2relation_line[local_idx] = r
                next_rel_idx += 1

        # 将答案实体映射到局部索引
        local_answer_entities = set()
        for ans_idx in answer_entities:
            if ans_idx in entity_idx2local:
                local_answer_entities.add(entity_idx2local[ans_idx])

        # 将话题实体映射到局部索引
        local_topic_entities = [entity_idx2local[idx] for idx in topic_entities if idx in entity_idx2local]

        if not local_topic_entities:
            return {"id": sample['id'], "error": "No valid topic entities"}

        # 检索路径
        num_nodes = node_features.size(0)
        all_paths = []

        # 修复：直接使用加载的完整 retriever，不再创建新的
        retriever = self.model['retriever']
        retriever.eval()

        with torch.no_grad():
            for start_node in local_topic_entities:
                if start_node >= num_nodes:
                    continue

                # 使用训练好的 retriever 进行束搜索
                # 编码子图（使用训练好的 GNN 层）
                encoded_features = retriever.encode_subgraph(node_features, edge_index, edge_types)

                # 束搜索
                result = retriever(
                    topic_entities=[start_node],
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_types=edge_types,
                    query_embedding=query_embedding,
                    entity_names=[f"entity_{i}" for i in range(num_nodes)],
                    top_k=self.config.top_k
                )

                paths = result['paths']
                similarities = result['similarities']

                for path, sim in zip(paths, similarities):
                    all_paths.append({
                        'nodes': path.nodes,
                        'relations': path.relations,
                        'similarity': float(sim),
                        'start_node': start_node
                    })

        # 按相似度排序
        all_paths.sort(key=lambda x: x['similarity'], reverse=True)
        top_k_paths = all_paths[:self.config.top_k]

        # 检查每个K值是否命中
        hits = {}
        for k in range(1, self.config.top_k + 1):
            hit = False
            for path in top_k_paths[:k]:
                # 检查路径是否包含答案实体
                path_nodes = set(path['nodes'])
                if path_nodes & local_answer_entities:  # 有交集即为命中
                    hit = True
                    break
            hits[f'Hit@{k}'] = hit

        # 将路径转换为实体名称和关系名称
        readable_paths = []
        for path in top_k_paths[:5]:  # 只转换前5条路径
            path_nodes = path['nodes']
            path_rels = path['relations']

            # 将局部索引转换为实体名称
            entity_names = []
            for local_idx in path_nodes:
                entity_line = local2entity_idx.get(local_idx, -1)
                if 0 <= entity_line < len(self.dataset.entities):
                    entity_names.append(self.dataset.entities[entity_line])
                else:
                    entity_names.append(f"entity_{entity_line}")

            # 将关系索引转换为关系名称
            rel_names = []
            for rel_local_idx in path_rels:
                # 从 local2relation_line 中找到原始关系行号
                rel_line = local2relation_line.get(rel_local_idx, rel_local_idx)
                rel_name = self.dataset.relations[rel_line] if rel_line < len(self.dataset.relations) else f"rel_{rel_line}"
                rel_names.append(rel_name)

            # 构建可读路径字符串
            path_str = entity_names[0]
            for i, rel_name in enumerate(rel_names):
                if i + 1 < len(entity_names):
                    path_str += f" --{rel_name}--> {entity_names[i + 1]}"

            readable_paths.append({
                'path_indices': path_nodes,
                'path_entities': entity_names,
                'path_relations': rel_names,
                'path_string': path_str,
                'similarity': path['similarity'],
                'contains_answer': bool(set(path_nodes) & local_answer_entities)
            })

        # 获取答案实体的名称
        answer_entity_names = []
        for ans_local in local_answer_entities:
            ans_line = local2entity_idx.get(ans_local, -1)
            if 0 <= ans_line < len(self.dataset.entities):
                answer_entity_names.append(self.dataset.entities[ans_line])

        # 获取话题实体的名称
        topic_entity_names = []
        for topic_local in local_topic_entities:
            topic_line = local2entity_idx.get(topic_local, -1)
            if 0 <= topic_line < len(self.dataset.entities):
                topic_entity_names.append(self.dataset.entities[topic_line])

        return {
            'id': sample['id'],
            'question': question,
            'num_answer_entities': len(local_answer_entities),
            'num_paths_found': len(all_paths),
            'hits': hits,
            'top_paths': top_k_paths[:5],  # 原始路径详情
            'readable_paths': readable_paths,  # 可读路径
            'answer_entity_names': answer_entity_names,  # 答案实体名称
            'topic_entity_names': topic_entity_names  # 话题实体名称
        }

    def evaluate(self) -> Tuple[Dict[str, float], List[Dict]]:
        """
        评估整个测试集

        返回:
            metrics: 包含Hit@1~Hit@10的字典
            error_samples: 错误样本列表
        """
        print(f"\nEvaluating on {len(self.dataset)} test samples...")

        # 初始化命中计数器
        hit_counts = {f'Hit@{k}': 0 for k in range(1, self.config.top_k + 1)}
        total_samples = 0
        valid_samples = 0
        error_samples = []
        detailed_results = []  # 收集详细结果用于保存

        # 限制样本数
        max_samples = len(self.dataset) if self.config.max_samples is None \
                      else min(self.config.max_samples, len(self.dataset))

        # 计算每 2% 的进度间隔
        progress_interval = max(1, max_samples // 50)  # 2% = 1/50
        last_progress = -1

        for idx in tqdm(range(max_samples), desc="Evaluating"):
            result = self.evaluate_sample(idx)

            if 'error' in result:
                error_samples.append({'idx': idx, 'error': result['error']})
                continue

            valid_samples += 1

            # 统计命中
            for k in range(1, self.config.top_k + 1):
                if result['hits'][f'Hit@{k}']:
                    hit_counts[f'Hit@{k}'] += 1

            total_samples += 1

            # 收集详细结果
            if 'readable_paths' in result:
                detailed_results.append({
                    'id': result['id'],
                    'question': result['question'],
                    'topic_entities': result.get('topic_entity_names', []),
                    'answer_entities': result.get('answer_entity_names', []),
                    'hit@1': result['hits'].get('Hit@1', False),
                    'hit@5': result['hits'].get('Hit@5', False),
                    'hit@10': result['hits'].get('Hit@10', False),
                    'num_paths_found': result['num_paths_found'],
                    'retrieved_paths': result['readable_paths']
                })

            # 每 2% 进度输出中间结果
            current_progress = (idx + 1) // progress_interval
            if current_progress > last_progress and valid_samples > 0:
                last_progress = current_progress
                progress_pct = min((idx + 1) * 100 // max_samples, 100)

                # 计算当前命中率
                print(f"\n[进度 {progress_pct}%] 中间结果 (已处理 {valid_samples} 个有效样本):")
                hit_rates = []
                for k in [1, 3, 5, 10]:
                    if k <= self.config.top_k:
                        hit_rate = hit_counts[f'Hit@{k}'] / valid_samples * 100
                        hit_rates.append(f"Hit@{k}: {hit_rate:5.2f}%")
                print("  " + " | ".join(hit_rates))

        # 计算命中率
        metrics = {}
        for k in range(1, self.config.top_k + 1):
            hit_rate = hit_counts[f'Hit@{k}'] / valid_samples if valid_samples > 0 else 0.0
            metrics[f'Hit@{k}'] = hit_rate * 100  # 转换为百分比

        metrics['total_samples'] = max_samples
        metrics['valid_samples'] = valid_samples
        metrics['error_samples'] = len(error_samples)

        return metrics, error_samples, detailed_results

    def run(self):
        """运行评估"""
        print("=" * 80)
        print("GraphRAG Test Evaluation")
        print("=" * 80)
        print(f"Device: {self.config.device}")
        print(f"Checkpoint: {self.config.checkpoint_path}")
        print(f"Top-K: {self.config.top_k}")

        # 加载测试集
        print("\nLoading test dataset...")
        dataset_config = GraphRAGConfig(
            graph_query_path=self.config.test_graph_query_path,
            original_data_path=self.config.test_original_data_path,
            entities_path=self.config.entities_path,
            relations_path=self.config.relations_path,
            num_relations=self.config.num_relations,
            bert_model_path=self.config.bert_model_path,
            top_k=self.config.top_k
        )
        dataset = CWQDataset(dataset_config)
        self.set_dataset(dataset)
        print(f"Test samples: {len(dataset)}")

        # 运行评估
        metrics, error_samples, detailed_results = self.evaluate()

        # 打印结果
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Valid samples: {metrics['valid_samples']}")
        print(f"Error samples: {metrics['error_samples']}")
        print("-" * 80)

        # 打印Hit@1~Hit@10
        for k in range(1, self.config.top_k + 1):
            hit_rate = metrics[f'Hit@{k}']
            print(f"Hit@{k:2d}: {hit_rate:6.2f}%")

        print("=" * 80)

        # 保存结果
        if self.config.output_path:
            output = {
                'config': {
                    'checkpoint_path': self.config.checkpoint_path,
                    'test_graph_query_path': self.config.test_graph_query_path,
                    'test_original_data_path': self.config.test_original_data_path,
                    'beam_width': self.config.beam_width,
                    'max_path_length': self.config.max_path_length,
                    'top_k': self.config.top_k
                },
                'metrics': metrics,
                'error_samples': error_samples[:20]  # 只保存前20个错误样本
            }

            Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {self.config.output_path}")

            # 保存详细路径结果
            if detailed_results:
                vis_path = str(Path(self.config.output_path).with_suffix('.paths.json'))
                with open(vis_path, 'w', encoding='utf-8') as f:
                    json.dump(detailed_results, f, indent=2, ensure_ascii=False)
                print(f"Detailed paths saved to {vis_path}")

        return metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GraphRAG Test Evaluation")

    # 数据路径
    parser.add_argument("--test_graph_query_path", type=str, required=True,
                        help="测试集图查询数据路径")
    parser.add_argument("--test_original_data_path", type=str, required=True,
                        help="测试集原始数据路径")
    parser.add_argument("--entities_path", type=str, required=True,
                        help="实体列表路径")
    parser.add_argument("--relations_path", type=str, required=True,
                        help="关系列表路径")

    # 模型配置
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="训练好的模型检查点路径")
    parser.add_argument("--bert_model_path", type=str, default="bert-base-uncased",
                        help="BERT模型路径")
    parser.add_argument("--hidden_dim", type=int, default=768,
                        help="隐藏层维度")
    parser.add_argument("--num_query_layers", type=int, default=3,
                        help="查询编码器层数")
    parser.add_argument("--num_retriever_layers", type=int, default=2,
                        help="检索器层数")
    parser.add_argument("--num_relations", type=int, default=100,
                        help="关系类型数")

    # 推理配置
    parser.add_argument("--beam_width", type=int, default=3,
                        help="束搜索宽度")
    parser.add_argument("--max_path_length", type=int, default=3,
                        help="最大路径长度")
    parser.add_argument("--similarity_threshold", type=float, default=0.0,
                        help="相似度阈值")
    parser.add_argument("--top_k", type=int, default=10,
                        help="计算Hit@K的最大K值")

    # 其他配置
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数（用于测试）")
    parser.add_argument("--output_path", type=str, default="output/test_results.json",
                        help="评估结果保存路径")

    return parser.parse_args()


def main():
    args = parse_args()

    config = EvalConfig(
        test_graph_query_path=args.test_graph_query_path,
        test_original_data_path=args.test_original_data_path,
        entities_path=args.entities_path,
        relations_path=args.relations_path,
        checkpoint_path=args.checkpoint_path,
        bert_model_path=args.bert_model_path,
        hidden_dim=args.hidden_dim,
        num_query_layers=args.num_query_layers,
        num_retriever_layers=args.num_retriever_layers,
        num_relations=args.num_relations,
        beam_width=args.beam_width,
        max_path_length=args.max_path_length,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
        device=args.device,
        max_samples=args.max_samples,
        output_path=args.output_path
    )

    evaluator = GraphRAGEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
