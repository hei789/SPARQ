#!/usr/bin/env python3
"""GraphRAG 整体程序入口 - 完整模型流程"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_query_encoder.models import GraphQueryEncoder
from graph_retriever import GraphRetriever, ReasoningPath


@dataclass
class GraphRAGConfig:
    """GraphRAG配置"""
    graph_query_path: str
    original_data_path: str
    entities_path: str
    relations_path: str
    hidden_dim: int = 768
    num_query_layers: int = 3
    num_retriever_layers: int = 2
    num_relations: int = 100
    beam_width: int = 3
    max_path_length: int = 3
    similarity_threshold: float = 0.0
    top_k: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples: Optional[int] = None
    output_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


class CWQDataset:
    """CWQ数据集加载器"""

    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.entities = self._load_entities()
        self.relations = self._load_relations()
        self.graph_queries = self._load_graph_queries()
        self.original_data = self._load_original_data()
        self.id_to_original = {item['id']: item for item in self.original_data}

    def _load_entities(self) -> List[str]:
        with open(self.config.entities_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def _load_relations(self) -> List[str]:
        with open(self.config.relations_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def _load_graph_queries(self) -> List[Dict]:
        with open(self.config.graph_query_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_original_data(self) -> List[Dict]:
        with open(self.config.original_data_path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f]

    def get_sample(self, idx: int) -> Optional[Dict]:
        if idx >= len(self.graph_queries):
            return None
        graph_query = self.graph_queries[idx]
        original_data = self.id_to_original.get(graph_query['id'], {})
        return {
            'id': graph_query['id'],
            'question': graph_query.get('question', ''),
            'triples': graph_query.get('triples', []),
            'answers': graph_query.get('answers', []),
            'subgraph': original_data.get('subgraph', {}),
            'entities': original_data.get('entities', [])
        }

    def __len__(self) -> int:
        return len(self.graph_queries)


class SubgraphProcessor:
    """子图处理器"""

    def __init__(self, entities: List[str], relations: List[str], hidden_dim: int, device: torch.device):
        self.entities = entities
        self.relations = relations
        self.hidden_dim = hidden_dim
        self.device = device

    def build_subgraph(self, subgraph_data: Dict, entity_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        """从原始数据构建PyG格式的子图"""
        tuples = subgraph_data.get('tuples', [])
        all_entity_indices = set()
        for h, r, t in tuples:
            all_entity_indices.add(h)
            all_entity_indices.add(t)

        entity_list = sorted(list(all_entity_indices))
        idx2local = {idx: i for i, idx in enumerate(entity_list)}
        num_nodes = len(entity_list)

        entity2idx = {}
        for idx in entity_list:
            if 0 <= idx < len(self.entities):
                entity2idx[self.entities[idx]] = idx2local[idx]

        node_features = torch.randn(num_nodes, self.hidden_dim, device=self.device)
        edges, edge_types, relation2idx = [], [], {}
        next_rel_idx = 0

        for h, r, t in tuples:
            src, dst = idx2local[h], idx2local[t]
            if r not in relation2idx:
                relation2idx[r] = next_rel_idx
                next_rel_idx += 1
            edges.append([src, dst])
            edge_types.append(relation2idx[r])

        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        edge_types = torch.tensor(edge_types, dtype=torch.long, device=self.device)
        return node_features, edge_index, edge_types, entity2idx

    def locate_topic_entities(self, triples: List[List[str]], entity2idx: Dict[str, int]) -> List[int]:
        """将图查询中的话题实体定位到子图上"""
        topic_entities = set()
        for head, rel, tail in triples:
            if head != "?":
                topic_entities.add(head)
            if tail != "?":
                topic_entities.add(tail)
        return [entity2idx[e] for e in topic_entities if e in entity2idx]


class GraphRAGPipeline(nn.Module):
    """GraphRAG完整流程"""

    def __init__(self, config: GraphRAGConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.query_encoder = GraphQueryEncoder(
            hidden_dim=config.hidden_dim,
            num_gnn_layers=config.num_query_layers,
            num_relations=config.num_relations,
            use_query_centered_pooling=True,
            use_bfs_gnn=True
        ).to(self.device)
        self.retriever = GraphRetriever(
            hidden_dim=config.hidden_dim,
            num_gnn_layers=config.num_retriever_layers,
            num_relations=config.num_relations,
            beam_width=config.beam_width,
            max_path_length=config.max_path_length,
            similarity_threshold=config.similarity_threshold
        ).to(self.device)
        self.subgraph_processor = None

    def set_dataset(self, dataset: CWQDataset):
        self.dataset = dataset
        self.subgraph_processor = SubgraphProcessor(
            entities=dataset.entities,
            relations=dataset.relations,
            hidden_dim=self.config.hidden_dim,
            device=self.device
        )

    def process_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset.get_sample(idx)
        if sample is None:
            return {"error": "Sample not found"}

        question, triples, subgraph_data, answers = sample['question'], sample['triples'], sample['subgraph'], sample['answers']

        if not subgraph_data or 'tuples' not in subgraph_data:
            return {"id": sample['id'], "question": question, "error": "No subgraph data"}

        self.query_encoder.eval()
        with torch.no_grad():
            query_result = self.query_encoder(triples, return_all_node_features=True)
            query_embedding = query_result[1] if isinstance(query_result, tuple) else query_result

        node_features, edge_index, edge_types, entity2idx = self.subgraph_processor.build_subgraph(subgraph_data, sample['entities'])
        topic_entities = self.subgraph_processor.locate_topic_entities(triples, entity2idx)

        if len(topic_entities) == 0:
            return {"id": sample['id'], "question": question, "error": "No topic entities found"}

        self.retriever.eval()
        with torch.no_grad():
            result = self.retriever(
                topic_entities=topic_entities,
                node_features=node_features,
                edge_index=edge_index,
                edge_types=edge_types,
                query_embedding=query_embedding,
                entity_names=list(entity2idx.keys()),
                top_k=self.config.top_k
            )

        paths, similarities = result['paths'], result['similarities']
        return {
            "id": sample['id'],
            "question": question,
            "triples": triples,
            "answers": answers,
            "num_topic_entities": len(topic_entities),
            "num_paths_found": len(paths),
            "paths": [{"nodes": p.nodes, "relations": p.relations,
                      "similarity": float(similarities[i]) if i < len(similarities) else 0.0}
                     for i, p in enumerate(paths[:self.config.top_k])]
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GraphRAG")
    parser.add_argument("--graph_query_path", type=str, required=True, help="图查询数据路径")
    parser.add_argument("--original_data_path", type=str, required=True, help="原始数据路径")
    parser.add_argument("--entities_path", type=str, required=True, help="实体列表路径")
    parser.add_argument("--relations_path", type=str, required=True, help="关系列表路径")
    parser.add_argument("--hidden_dim", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--num_query_layers", type=int, default=3, help="查询编码器层数")
    parser.add_argument("--num_retriever_layers", type=int, default=2, help="检索器层数")
    parser.add_argument("--num_relations", type=int, default=100, help="关系类型数")
    parser.add_argument("--beam_width", type=int, default=3, help="束搜索宽度")
    parser.add_argument("--max_path_length", type=int, default=3, help="最大路径长度")
    parser.add_argument("--similarity_threshold", type=float, default=0.0, help="相似度阈值")
    parser.add_argument("--top_k", type=int, default=10, help="返回路径数")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--output_path", type=str, default="output/results.json", help="输出路径")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="检查点路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("GraphRAG")
    print("=" * 80)

    config = GraphRAGConfig(
        graph_query_path=args.graph_query_path,
        original_data_path=args.original_data_path,
        entities_path=args.entities_path,
        relations_path=args.relations_path,
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
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path
    )

    print("\n加载数据...")
    try:
        dataset = CWQDataset(config)
        print(f"  样本数: {len(dataset.graph_queries)}, 实体数: {len(dataset.entities)}")
    except FileNotFoundError as e:
        print(f"数据加载失败: {e}")
        return

    print("初始化模型...")
    pipeline = GraphRAGPipeline(config)
    pipeline.set_dataset(dataset)

    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        if 'query_encoder_state' in checkpoint:
            pipeline.query_encoder.load_state_dict(checkpoint['query_encoder_state'])
        if 'retriever_state' in checkpoint:
            pipeline.retriever.load_state_dict(checkpoint['retriever_state'])

    num_samples = min(len(dataset), config.max_samples or len(dataset))
    print(f"\n处理 {num_samples} 个样本...")

    all_results = []
    for i in range(num_samples):
        if i % 10 == 0:
            print(f"  {i+1}/{num_samples}")
        result = pipeline.process_sample(i)
        all_results.append(result)

    success_count = sum(1 for r in all_results if 'error' not in r)
    total_paths = sum(r.get('num_paths_found', 0) for r in all_results if 'error' not in r)
    print(f"\n完成: 成功{success_count}/{num_samples}, 总路径{total_paths}")

    if config.output_path:
        Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config.output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"结果保存: {config.output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
