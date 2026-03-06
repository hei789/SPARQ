"""
图神经网络检索器
基于GNN的推理路径检索，支持束搜索和相关度匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import heapq

from .gnn_layers import RGCNLayer


@dataclass
class ReasoningPath:
    """推理路径数据类"""
    nodes: List[int]          # 节点索引序列
    relations: List[int]      # 关系索引序列
    score: float              # 路径得分
    query_similarity: float   # 与查询的相似度

    def __lt__(self, other):
        """用于堆排序"""
        return self.score > other.score  # 大根堆


class PathEncoder(nn.Module):
    """
    推理路径编码器
    将路径编码为向量表示
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_relations: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 关系嵌入
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        # 路径编码器（使用LSTM或Transformer）
        self.path_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # 节点特征 + 关系特征
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        path_nodes: List[int],
        path_relations: List[int]
    ) -> torch.Tensor:
        """
        编码单个路径

        Args:
            node_features: 所有节点的特征 (num_nodes, hidden_dim)
            path_nodes: 路径上的节点索引
            path_relations: 路径上的关系索引

        Returns:
            path_embedding: 路径的向量表示 (hidden_dim,)
        """
        if len(path_nodes) == 0:
            return torch.zeros(self.hidden_dim, device=node_features.device)

        # 获取路径上的节点特征
        path_node_feats = node_features[path_nodes]  # (path_len, hidden_dim)

        # 获取关系特征
        rel_indices = torch.tensor(path_relations, device=node_features.device)
        path_rel_feats = self.relation_embedding(rel_indices)  # (path_len-1, hidden_dim)

        # 对于起始节点，使用零向量作为关系特征
        zero_rel = torch.zeros(1, self.hidden_dim, device=node_features.device)
        if len(path_relations) < len(path_nodes):
            path_rel_feats = torch.cat([zero_rel, path_rel_feats], dim=0)

        # 拼接节点和关系特征
        path_feats = torch.cat([path_node_feats, path_rel_feats], dim=-1)  # (path_len, hidden_dim*2)

        # LSTM编码
        path_feats = path_feats.unsqueeze(0)  # (1, path_len, hidden_dim*2)
        lstm_out, (h_n, c_n) = self.path_lstm(path_feats)

        # 使用最终的隐藏状态
        path_embedding = torch.cat([h_n[0], h_n[1]], dim=-1)  # (hidden_dim*2,)
        path_embedding = self.output_projection(path_embedding)

        return path_embedding


class GraphRetriever(nn.Module):
    """
    图神经网络检索器

    功能：
    1. 从话题实体出发进行束搜索，检索推理路径
    2. 使用GNN编码子图节点特征
    3. 计算路径与图查询的相关度
    4. 返回高相关度的推理路径
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_gnn_layers: int = 2,
        num_relations: int = 100,
        dropout: float = 0.1,
        beam_width: int = 3,
        max_path_length: int = 3,
        similarity_threshold: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beam_width = beam_width
        self.max_path_length = max_path_length
        self.similarity_threshold = similarity_threshold

        # GNN用于编码子图
        self.gnn_layers = nn.ModuleList([
            RGCNLayer(
                in_dim=hidden_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                num_relations=num_relations,
                dropout=dropout
            )
            for i in range(num_gnn_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # 路径编码器
        self.path_encoder = PathEncoder(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            dropout=dropout
        )

        # 查询-路径相似度计算
        self.similarity_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 关系到索引的映射
        self.relation2idx: Dict[str, int] = {}
        self.next_relation_idx = 0

    def _get_or_create_relation_idx(self, relation: str) -> int:
        """获取或创建关系的索引"""
        if relation not in self.relation2idx:
            self.relation2idx[relation] = self.next_relation_idx
            self.next_relation_idx += 1
        return self.relation2idx[relation]

    def encode_subgraph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """
        使用GNN编码子图

        Args:
            node_features: 节点初始特征 (num_nodes, hidden_dim)
            edge_index: 边索引 (2, num_edges)
            edge_types: 边类型 (num_edges,)

        Returns:
            encoded_features: 编码后的节点特征 (num_nodes, hidden_dim)
        """
        x = node_features
        for layer, norm in zip(self.gnn_layers, self.layer_norms):
            x = layer(x, edge_index, edge_types)
            x = norm(x)
        return x

    def compute_path_query_similarity(
        self,
        path_embedding: torch.Tensor,
        query_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        计算路径与查询的相似度

        Args:
            path_embedding: 路径嵌入 (hidden_dim,)
            query_embedding: 查询嵌入 (hidden_dim,)

        Returns:
            similarity: 相似度分数 (标量)
        """
        combined = torch.cat([path_embedding, query_embedding], dim=-1)
        score = self.similarity_scorer(combined)
        return torch.sigmoid(score).squeeze()

    def beam_search_paths(
        self,
        start_node: int,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        query_embedding: torch.Tensor,
        entity_names: List[str]
    ) -> List[ReasoningPath]:
        """
        从起始节点进行束搜索，检索推理路径

        Args:
            start_node: 起始节点索引（话题实体）
            node_features: 节点特征 (num_nodes, hidden_dim)
            edge_index: 边索引 (2, num_edges)
            edge_types: 边类型 (num_edges,)
            query_embedding: 图查询嵌入 (hidden_dim,)
            entity_names: 实体名称列表（用于调试）

        Returns:
            paths: 检索到的推理路径列表
        """
        device = node_features.device
        num_nodes = node_features.size(0)

        # 构建邻接表
        adj_list = [[] for _ in range(num_nodes)]
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        edge_type_list = edge_types.tolist()

        for s, d, et in zip(src, dst, edge_type_list):
            adj_list[s].append((d, et))  # (邻居节点, 关系类型)
            adj_list[d].append((s, et))  # 无向图

        # 束搜索
        # 当前保留的候选路径 [(score, path_nodes, path_relations), ...]
        beams = [(0.0, [start_node], [])]
        completed_paths = []

        for step in range(self.max_path_length):
            candidates = []

            for score, path_nodes, path_relations in beams:
                current_node = path_nodes[-1]

                # 扩展当前路径
                for next_node, rel_type in adj_list[current_node]:
                    # 避免环路
                    if next_node in path_nodes:
                        continue

                    new_path_nodes = path_nodes + [next_node]
                    new_path_relations = path_relations + [rel_type]

                    # 编码新路径
                    path_embedding = self.path_encoder(
                        node_features, new_path_nodes, new_path_relations
                    )

                    # 计算与查询的相似度
                    similarity = self.compute_path_query_similarity(
                        path_embedding, query_embedding
                    )

                    # 综合得分：路径长度惩罚 + 查询相似度
                    path_score = similarity.item() - 0.1 * len(new_path_nodes)

                    candidates.append((
                        path_score,
                        new_path_nodes,
                        new_path_relations,
                        similarity.item()
                    ))

            if len(candidates) == 0:
                break

            # 选择Top-K路径
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = candidates[:self.beam_width]

            # 更新beams
            beams = [(c[0], c[1], c[2]) for c in top_candidates]

            # 保存达到长度的路径
            for score, p_nodes, p_rels, sim in top_candidates:
                if sim >= self.similarity_threshold:
                    completed_paths.append(ReasoningPath(
                        nodes=p_nodes,
                        relations=p_rels,
                        score=score,
                        query_similarity=sim
                    ))

        # 添加beams中剩余的路径
        for score, path_nodes, path_relations in beams:
            path_embedding = self.path_encoder(
                node_features, path_nodes, path_relations
            )
            similarity = self.compute_path_query_similarity(
                path_embedding, query_embedding
            )

            if similarity.item() >= self.similarity_threshold:
                completed_paths.append(ReasoningPath(
                    nodes=path_nodes,
                    relations=path_relations,
                    score=score,
                    query_similarity=similarity.item()
                ))

        # 去重并排序
        unique_paths = self._deduplicate_paths(completed_paths)
        unique_paths.sort(key=lambda x: x.query_similarity, reverse=True)

        return unique_paths

    def _deduplicate_paths(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """去重路径"""
        seen = set()
        unique = []
        for path in paths:
            path_tuple = (tuple(path.nodes), tuple(path.relations))
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique.append(path)
        return unique

    def forward(
        self,
        topic_entities: List[int],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        query_embedding: torch.Tensor,
        entity_names: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, any]:
        """
        前向传播：检索相关推理路径

        Args:
            topic_entities: 话题实体索引列表
            node_features: 节点初始特征 (num_nodes, hidden_dim)
            edge_index: 边索引 (2, num_edges)
            edge_types: 边类型 (num_edges,)
            query_embedding: 图查询嵌入 (hidden_dim,)
            entity_names: 实体名称列表（可选，用于调试）
            top_k: 返回的路径数量

        Returns:
            result: 包含以下字段的字典
                - paths: 检索到的推理路径列表
                - path_embeddings: 路径嵌入
                - similarities: 路径与查询的相似度
                - topic_entity_paths: 每个话题实体对应的路径
        """
        device = node_features.device

        if entity_names is None:
            entity_names = [f"entity_{i}" for i in range(node_features.size(0))]

        # 1. 使用GNN编码子图
        encoded_features = self.encode_subgraph(
            node_features, edge_index, edge_types
        )

        # 2. 从每个话题实体进行束搜索
        all_paths = []
        topic_entity_paths = {}

        for topic_entity in topic_entities:
            paths = self.beam_search_paths(
                start_node=topic_entity,
                node_features=encoded_features,
                edge_index=edge_index,
                edge_types=edge_types,
                query_embedding=query_embedding,
                entity_names=entity_names
            )

            topic_entity_paths[topic_entity] = paths
            all_paths.extend(paths)

        # 3. 全局排序并选择Top-K
        all_paths.sort(key=lambda x: x.query_similarity, reverse=True)
        top_paths = all_paths[:top_k]

        # 4. 编码最终选择的路径
        path_embeddings = []
        for path in top_paths:
            emb = self.path_encoder(
                encoded_features, path.nodes, path.relations
            )
            path_embeddings.append(emb)

        if len(path_embeddings) > 0:
            path_embeddings = torch.stack(path_embeddings, dim=0)
        else:
            path_embeddings = torch.zeros(0, self.hidden_dim, device=device)

        return {
            "paths": top_paths,
            "path_embeddings": path_embeddings,
            "similarities": torch.tensor(
                [p.query_similarity for p in top_paths],
                device=device
            ) if len(top_paths) > 0 else torch.zeros(0, device=device),
            "topic_entity_paths": topic_entity_paths,
            "encoded_node_features": encoded_features
        }


class IntegratedRetriever(nn.Module):
    """
    集成检索器
    将图查询编码器和检索器结合起来
    """

    def __init__(
        self,
        query_encoder: nn.Module,
        retriever: GraphRetriever,
        device: torch.device = None
    ):
        super().__init__()
        self.query_encoder = query_encoder
        self.retriever = retriever
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_topic_entities(
        self,
        triples: List[List[str]]
    ) -> List[str]:
        """
        从三元组中提取话题实体（非"?"的节点）

        Args:
            triples: 三元组列表 [[head, relation, tail], ...]

        Returns:
            topic_entities: 话题实体名称列表
        """
        entities = set()
        for head, rel, tail in triples:
            if head != "?":
                entities.add(head)
            if tail != "?":
                entities.add(tail)
        return list(entities)

    def forward(
        self,
        triples: List[List[str]],
        subgraph_node_features: torch.Tensor,
        subgraph_edge_index: torch.Tensor,
        subgraph_edge_types: torch.Tensor,
        entity2idx: Dict[str, int],
        top_k: int = 10
    ) -> Dict[str, any]:
        """
        集成前向传播

        Args:
            triples: 图查询的三元组列表
            subgraph_node_features: 子图节点特征
            subgraph_edge_index: 子图边索引
            subgraph_edge_types: 子图边类型
            entity2idx: 实体到索引的映射
            top_k: 返回的路径数量

        Returns:
            result: 包含查询嵌入、检索路径等信息的字典
        """
        # 1. 编码图查询
        query_result = self.query_encoder(
            triples,
            return_all_node_features=True
        )

        if isinstance(query_result, tuple):
            query_node_features, query_embedding, query_entity2idx, query_idx = query_result
        else:
            query_embedding = query_result
            query_node_features = None
            query_entity2idx = {}
            query_idx = -1

        # 2. 提取话题实体
        topic_entity_names = self.extract_topic_entities(triples)

        # 3. 将话题实体映射到子图索引
        # 使用相似度匹配或精确匹配
        topic_entities = []
        for entity_name in topic_entity_names:
            if entity_name in entity2idx:
                topic_entities.append(entity2idx[entity_name])
            # TODO: 实现模糊匹配

        if len(topic_entities) == 0:
            # 如果没有找到话题实体，使用子图中所有节点
            topic_entities = list(range(subgraph_node_features.size(0)))

        # 4. 使用检索器检索路径
        entity_names_list = list(entity2idx.keys())
        retrieval_result = self.retriever(
            topic_entities=topic_entities,
            node_features=subgraph_node_features,
            edge_index=subgraph_edge_index,
            edge_types=subgraph_edge_types,
            query_embedding=query_embedding,
            entity_names=entity_names_list,
            top_k=top_k
        )

        return {
            "query_embedding": query_embedding,
            "topic_entities": topic_entities,
            "topic_entity_names": topic_entity_names,
            **retrieval_result
        }
