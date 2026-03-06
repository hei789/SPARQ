"""
图结构查询编码器
整合BERT初始编码和GNN消息传递，输出以"?"为中心的图结构表征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import json

from .embedding_layer import EmbeddingLayer
from .gnn_layers import MultiLayerGNN, MultiLayerQueryCenteredGNN


def compute_bfs_distances(
    edge_index: torch.Tensor,
    query_idx: int,
    num_nodes: int
) -> Tuple[torch.Tensor, int]:
    """
    计算每个节点到"?"节点的BFS距离

    Args:
        edge_index: 边索引 (2, num_edges)
        query_idx: "?"节点的索引
        num_nodes: 节点总数

    Returns:
        distances: 每个节点的BFS距离 (num_nodes,)
        max_distance: 最大距离
    """
    device = edge_index.device
    distances = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

    if query_idx < 0 or query_idx >= num_nodes:
        return distances, 0

    # "?"节点距离为0
    distances[query_idx] = 0

    # BFS队列
    visited = {query_idx}
    queue = [(query_idx, 0)]

    # 构建邻接表（无向图）
    adj_list = [[] for _ in range(num_nodes)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()

    for s, d in zip(src, dst):
        adj_list[s].append(d)
        adj_list[d].append(s)  # 无向图，添加反向边

    # BFS遍历
    while queue:
        node, dist = queue.pop(0)

        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))

    # 未访问的节点（孤立节点）设置为最大距离+1
    max_distance = distances.max().item()
    distances[distances == -1] = max_distance + 1

    return distances, distances.max().item()


class GraphQueryEncoder(nn.Module):
    """
    图结构查询编码器

    输入: 图结构查询（三元组列表）
    输出: 以"?"为中心的图结构表征 h_eq

    使用反向BFS消息传递：从远离"?"的节点逐层向"?"聚合
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_gnn_layers: int = 3,
        num_relations: int = 100,
        bert_model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
        use_query_centered_pooling: bool = True,
        use_bfs_gnn: bool = True  # 新增：是否使用BFS定向GNN
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.use_query_centered_pooling = use_query_centered_pooling
        self.use_bfs_gnn = use_bfs_gnn

        # 1. 初始特征编码层（BERT）
        self.embedding_layer = EmbeddingLayer(
            bert_model_name=bert_model_name,
            hidden_dim=hidden_dim
        )

        # 2. 多层GNN（根据配置选择）
        if use_bfs_gnn:
            # 使用以"?"为中心的反向BFS GNN
            self.gnn = MultiLayerQueryCenteredGNN(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_gnn_layers,
                num_relations=num_relations,
                dropout=dropout
            )
        else:
            # 使用标准GNN
            self.gnn = MultiLayerGNN(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_gnn_layers,
                num_relations=num_relations,
                dropout=dropout
            )

        # 3. 关系嵌入表（用于边类型索引）
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        # 4. 输出层：将GNN输出整合为最终的图查询表征
        if use_query_centered_pooling:
            # 以"?"节点为中心的表征
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            # 全局池化
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # 关系到索引的映射（动态构建）
        self.relation2idx: Dict[str, int] = {}
        self.next_relation_idx = 0

    def _get_or_create_relation_idx(self, relation: str) -> int:
        """获取或创建关系的索引"""
        if relation not in self.relation2idx:
            self.relation2idx[relation] = self.next_relation_idx
            self.next_relation_idx += 1
        return self.relation2idx[relation]

    def _build_graph(
        self,
        triples: List[List[str]],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int], int]:
        """
        从三元组列表构建图结构

        Args:
            triples: 三元组列表 [[head, relation, tail], ...]
            device: 计算设备

        Returns:
            node_features: 节点初始特征 (num_nodes, hidden_dim)
            edge_index: 边索引 (2, num_edges)
            edge_types: 边类型 (num_edges,)
            entity2idx: 实体到索引的映射
            query_idx: "?"节点的索引
        """
        # 收集所有唯一实体和关系
        entities = set()
        relations = set()
        query_idx = -1

        for head, rel, tail in triples:
            entities.add(head)
            entities.add(tail)
            relations.add(rel)

        # 创建实体到索引的映射
        entity_list = sorted(list(entities))
        entity2idx = {e: i for i, e in enumerate(entity_list)}

        # 找到"?"的索引
        if "?" in entity2idx:
            query_idx = entity2idx["?"]

        num_nodes = len(entity_list)

        # 编码节点特征
        node_features = self.embedding_layer.encode_entities(entity_list, device)

        # 构建边索引和边类型
        edge_list = []
        edge_type_list = []

        for head, rel, tail in triples:
            src_idx = entity2idx[head]
            dst_idx = entity2idx[tail]
            rel_idx = self._get_or_create_relation_idx(rel)

            # 添加边（双向图，但这里保持方向性）
            edge_list.append([src_idx, dst_idx])
            edge_type_list.append(rel_idx)

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
            edge_types = torch.tensor(edge_type_list, dtype=torch.long, device=device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_types = torch.zeros(0, dtype=torch.long, device=device)

        return node_features, edge_index, edge_types, entity2idx, query_idx

    def _pool_query_representation(
        self,
        node_features: torch.Tensor,
        query_idx: int
    ) -> torch.Tensor:
        """
        池化得到图查询表征

        Args:
            node_features: GNN输出的节点特征 (num_nodes, hidden_dim)
            query_idx: "?"节点的索引

        Returns:
            h_eq: 图查询表征 (hidden_dim,)
        """
        # 以"?"节点为中心：结合"?"节点表征和全局平均池化
        global_feature = node_features.mean(dim=0)  # (hidden_dim,)

        if self.use_query_centered_pooling and query_idx >= 0:
            query_feature = node_features[query_idx]  # (hidden_dim,)
            combined = torch.cat([query_feature, global_feature], dim=-1)  # (hidden_dim * 2,)
        else:
            # 如果没有找到"?"节点，使用全局特征拼接自身（保持维度一致）
            combined = torch.cat([global_feature, global_feature], dim=-1)  # (hidden_dim * 2,)

        h_eq = self.output_projection(combined)

        return h_eq

    def forward(
        self,
        triples: List[List[str]],
        return_all_node_features: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            triples: 三元组列表 [[head, relation, tail], ...]
            return_all_node_features: 是否返回所有节点的特征

        Returns:
            h_eq: 图查询表征 (hidden_dim,)
            或 (all_node_features, h_eq) 如果return_all_node_features为True
        """
        device = next(self.parameters()).device

        # 1. 构建图
        node_features, edge_index, edge_types, entity2idx, query_idx = self._build_graph(
            triples, device
        )

        # 2. GNN消息传递
        if self.use_bfs_gnn:
            # 计算BFS距离
            num_nodes = node_features.size(0)
            distances, max_distance = compute_bfs_distances(
                edge_index, query_idx, num_nodes
            )
            # 使用反向BFS定向GNN
            updated_features = self.gnn(
                node_features, edge_index, edge_types, distances, max_distance
            )
        else:
            # 使用标准GNN
            updated_features = self.gnn(node_features, edge_index, edge_types)

        # 3. 池化得到查询表征
        h_eq = self._pool_query_representation(updated_features, query_idx)

        if return_all_node_features:
            return updated_features, h_eq, entity2idx, query_idx

        return h_eq

    def encode_batch(
        self,
        batch_triples: List[List[List[str]]]
    ) -> torch.Tensor:
        """
        批量编码（对每个查询分别编码后堆叠）

        Args:
            batch_triples: 批次的三元组列表

        Returns:
            h_eq_batch: (batch_size, hidden_dim)
        """
        h_eq_list = []
        for triples in batch_triples:
            h_eq = self.forward(triples)
            h_eq_list.append(h_eq)

        return torch.stack(h_eq_list, dim=0)
