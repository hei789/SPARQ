"""
数据加载器模块
支持CWQ数据集加载和子图构建
"""

import json
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path


class CWQDataLoader:
    """CWQ数据加载器"""

    def __init__(
        self,
        data_path: str,
        entities_path: str,
        relations_path: str
    ):
        """
        初始化CWQ数据加载器

        Args:
            data_path: 图查询数据文件路径 (如: path/to/graph_query_dev.json)
            entities_path: 实体列表文件路径 (如: path/to/entities.txt)
            relations_path: 关系列表文件路径 (如: path/to/relations.txt)
        """
        self.data_path = data_path
        self.entities_path = entities_path
        self.relations_path = relations_path

        # 加载实体和关系映射
        self.entity_list = self._load_entities()
        self.relation_list = self._load_relations()

    def _load_entities(self) -> List[str]:
        """加载实体列表"""
        entities = []
        with open(self.entities_path, 'r', encoding='utf-8') as f:
            for line in f:
                entities.append(line.strip())
        return entities

    def _load_relations(self) -> List[str]:
        """加载关系列表"""
        relations = []
        with open(self.relations_path, 'r', encoding='utf-8') as f:
            for line in f:
                relations.append(line.strip())
        return relations

    def load_graph_queries(self) -> List[Dict[str, Any]]:
        """
        加载图查询数据

        Returns:
            图查询样本列表
        """
        file_path = Path(self.data_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def load_original_data(self, split: str = "dev") -> List[Dict[str, Any]]:
        """
        加载原始数据（包含子图）

        Args:
            split: 数据集划分 (train/dev/test)

        Returns:
            原始样本列表（JSONL格式，逐行读取）
        """
        # 从data_path推断原始数据路径
        # 假设原始数据在相同目录下，文件名为 {split}_simple.json
        data_dir = Path(self.data_path).parent
        file_path = data_dir / f"{split}_simple.json"

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        return data

    def get_entity_id_by_index(self, idx: int) -> str:
        """通过索引获取实体ID"""
        if 0 <= idx < len(self.entity_list):
            return self.entity_list[idx]
        return f"unknown_entity_{idx}"

    def get_relation_id_by_index(self, idx: int) -> str:
        """通过索引获取关系ID"""
        if 0 <= idx < len(self.relation_list):
            return self.relation_list[idx]
        return f"unknown_relation_{idx}"


class SubgraphBuilder:
    """子图构建器 - 将原始数据中的子图转换为PyG格式"""

    @staticmethod
    def build_from_tuples(
        tuples: List[List[int]],
        entity_indices: List[int],
        hidden_dim: int,
        device: torch.device
    ) -> tuple:
        """
        从tuples构建子图

        Args:
            tuples: 三元组列表 [[head_idx, rel_idx, tail_idx], ...]
            entity_indices: 实体索引列表
            hidden_dim: 隐藏维度
            device: 计算设备

        Returns:
            node_features, edge_index, edge_types, entity2idx
        """
        # 收集所有实体
        all_entities = set()
        for h, r, t in tuples:
            all_entities.add(h)
            all_entities.add(t)

        # 创建实体到索引的映射
        entity_list = sorted(list(all_entities))
        entity2idx = {e: i for i, e in enumerate(entity_list)}
        num_nodes = len(entity_list)

        # 节点特征（实际使用时应该用预训练编码）
        node_features = torch.randn(num_nodes, hidden_dim, device=device)

        # 构建边
        edges = []
        edge_types = []

        for h, r, t in tuples:
            src = entity2idx[h]
            dst = entity2idx[t]
            edges.append([src, dst])
            edge_types.append(r)

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        edge_types = torch.tensor(edge_types, dtype=torch.long, device=device)

        return node_features, edge_index, edge_types, entity2idx

    @staticmethod
    def build_from_entity_names(
        triples: List[List[str]],
        entity2idx: Dict[str, int],
        hidden_dim: int,
        device: torch.device
    ) -> tuple:
        """
        从实体名称三元组构建子图

        Args:
            triples: 三元组列表 [[head, relation, tail], ...]
            entity2idx: 实体名称到索引的映射
            hidden_dim: 隐藏维度
            device: 计算设备

        Returns:
            node_features, edge_index, edge_types, entity2idx
        """
        # 收集所有实体
        all_entities = set()
        for h, rel, t in triples:
            all_entities.add(h)
            all_entities.add(t)

        # 创建局部索引映射
        entity_list = sorted(list(all_entities))
        local_entity2idx = {e: i for i, e in enumerate(entity_list)}
        num_nodes = len(entity_list)

        # 节点特征
        node_features = torch.randn(num_nodes, hidden_dim, device=device)

        # 构建边
        edges = []
        edge_types = []

        # 关系到索引的映射（动态构建）
        relation2idx = {}
        next_rel_idx = 0

        for h, rel, t in triples:
            src = local_entity2idx[h]
            dst = local_entity2idx[t]

            if rel not in relation2idx:
                relation2idx[rel] = next_rel_idx
                next_rel_idx += 1

            edges.append([src, dst])
            edge_types.append(relation2idx[rel])

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        edge_types = torch.tensor(edge_types, dtype=torch.long, device=device)

        return node_features, edge_index, edge_types, local_entity2idx
