"""
GNN层实现
使用R-GCN（关系图卷积网络）处理带关系的图结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class RGCNLayer(nn.Module):
    """
    关系图卷积网络层
    对每个关系类型使用不同的权重矩阵进行消息传递
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        dropout: float = 0.1,
        use_basis: bool = False,
        num_basis: int = 10
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.use_basis = use_basis

        if use_basis:
            # 使用基分解减少参数
            self.num_basis = num_basis
            self.basis_weights = nn.Parameter(torch.randn(num_basis, in_dim, out_dim))
            self.alpha = nn.Parameter(torch.randn(num_relations, num_basis))
        else:
            # 为每种关系定义独立的权重矩阵
            self.relation_weights = nn.Parameter(
                torch.randn(num_relations, in_dim, out_dim)
            )

        # 自环连接的权重（节点自身的信息）
        self.self_weight = nn.Parameter(torch.randn(in_dim, out_dim))

        # 偏置和dropout
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.self_weight)
        if self.use_basis:
            nn.init.xavier_uniform_(self.basis_weights.view(self.num_basis, -1))
        else:
            nn.init.xavier_uniform_(self.relation_weights.view(self.num_relations, -1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 (num_nodes, in_dim)
            edge_index: 边索引 (2, num_edges)，[source, target]
            edge_types: 边类型 (num_edges,)

        Returns:
            out: 更新后的节点特征 (num_nodes, out_dim)
        """
        num_nodes = x.size(0)

        # 1. 自环连接
        self_message = torch.mm(x, self.self_weight)  # (num_nodes, out_dim)

        # 2. 邻居消息聚合
        neighbor_message = torch.zeros(num_nodes, self.out_dim, device=x.device)

        if edge_index.size(1) > 0:
            # 计算每种关系的权重矩阵
            if self.use_basis:
                # 基分解: W_r = sum_k alpha_{r,k} * B_k
                weights = torch.einsum('rb,bio->rio', self.alpha, self.basis_weights)
            else:
                weights = self.relation_weights

            # 消息传递
            src, dst = edge_index[0], edge_index[1]

            for r in range(self.num_relations):
                # 找到当前关系类型的边
                mask = edge_types == r
                if mask.sum() == 0:
                    continue

                # 源节点特征
                src_nodes = src[mask]
                src_features = x[src_nodes]  # (num_edges_r, in_dim)

                # 应用关系特定的权重
                messages = torch.mm(src_features, weights[r])  # (num_edges_r, out_dim)

                # 聚合到目标节点
                dst_nodes = dst[mask]
                neighbor_message.index_add_(0, dst_nodes, messages)

        # 归一化（可选：按度归一化）
        # neighbor_message = self._normalize_by_degree(neighbor_message, edge_index, num_nodes)

        # 合并自环和邻居消息
        out = self_message + neighbor_message + self.bias
        out = self.dropout(out)
        out = F.relu(out)

        return out

    def _normalize_by_degree(
        self,
        message: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """按节点度进行归一化"""
        dst = edge_index[1]
        degree = torch.bincount(dst, minlength=num_nodes).float().clamp(min=1)
        degree = degree.unsqueeze(1)  # (num_nodes, 1)
        return message / degree


class MultiLayerGNN(nn.Module):
    """
    多层GNN堆叠
    从最外层向内层逐步聚合信息
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_relations: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers

        # 构建GNN层
        self.layers = nn.ModuleList()

        # 第一层: input_dim -> hidden_dim
        self.layers.append(
            RGCNLayer(input_dim, hidden_dim, num_relations, dropout)
        )

        # 中间层: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(
                RGCNLayer(hidden_dim, hidden_dim, num_relations, dropout)
            )

        # 最后一层: hidden_dim -> output_dim
        if num_layers > 1:
            self.layers.append(
                RGCNLayer(hidden_dim, output_dim, num_relations, dropout)
            )

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 初始节点特征 (num_nodes, input_dim)
            edge_index: 边索引 (2, num_edges)
            edge_types: 边类型 (num_edges,)

        Returns:
            x: 最终的节点表征 (num_nodes, output_dim)
        """
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            x = layer(x, edge_index, edge_types)
            x = norm(x)

        return x


class QueryCenteredBFSLayer(nn.Module):
    """
    以"?"为中心的反向BFS消息传递层

    消息传递方向：从远离"?"的节点向"?"逐层收敛
    第k层：距离"?"为k的节点将信息传递给距离为k-1的节点
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # 为每种关系定义独立的权重矩阵
        self.relation_weights = nn.Parameter(
            torch.randn(num_relations, in_dim, out_dim)
        )

        # 自环连接的权重
        self.self_weight = nn.Parameter(torch.randn(in_dim, out_dim))

        # 偏置和dropout
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.self_weight)
        nn.init.xavier_uniform_(self.relation_weights.view(self.num_relations, -1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        distances: torch.Tensor,
        current_distance: int
    ) -> torch.Tensor:
        """
        前向传播 - 只处理特定BFS距离的节点

        Args:
            x: 节点特征 (num_nodes, in_dim)
            edge_index: 边索引 (2, num_edges)，[source, target]
            edge_types: 边类型 (num_edges,)
            distances: 每个节点到"?"的BFS距离 (num_nodes,)
            current_distance: 当前处理的距离层

        Returns:
            out: 更新后的节点特征 (num_nodes, out_dim)
        """
        num_nodes = x.size(0)
        device = x.device

        # 创建输出特征（复制输入，保持未更新节点不变）
        out = x.clone()

        # 1. 自环连接（所有节点都更新自身）
        self_message = torch.mm(x, self.self_weight)

        # 找到距离为 current_distance 的节点（信息源）
        source_mask = distances == current_distance
        if source_mask.sum() == 0:
            # 没有当前距离的节点，只更新自环
            out = x + self_message
            return F.relu(out + self.bias)

        # 2. 反向BFS消息传递：从距离d的节点传递给距离d-1的节点
        # 构建反向边索引（从target指向source）
        src, dst = edge_index[0], edge_index[1]

        # 对于每个源节点（距离为current_distance），找到它的邻居（距离为current_distance-1）
        neighbor_message = torch.zeros(num_nodes, self.out_dim, device=device)

        for r in range(self.num_relations):
            mask = edge_types == r
            if mask.sum() == 0:
                continue

            # 当前关系的边
            r_src = src[mask]
            r_dst = dst[mask]

            # 找到源节点是current_distance，目标节点是current_distance-1的边
            src_distances = distances[r_src]
            dst_distances = distances[r_dst]

            # 反向BFS：信息从距离d流向距离d-1
            # 即：源节点距离=current_distance，目标节点距离=current_distance-1
            valid_edges = (src_distances == current_distance) & \
                         (dst_distances == current_distance - 1)

            if valid_edges.sum() == 0:
                continue

            # 获取有效的源节点和目标节点
            valid_src = r_src[valid_edges]
            valid_dst = r_dst[valid_edges]

            # 源节点特征
            src_features = x[valid_src]

            # 应用关系特定的权重
            messages = torch.mm(src_features, self.relation_weights[r])

            # 聚合到目标节点（距离current_distance-1的节点）
            neighbor_message.index_add_(0, valid_dst, messages)

        # 3. 更新特征
        # 距离为 current_distance-1 的节点接收消息
        target_mask = distances == current_distance - 1

        # 所有节点都应用自环
        out = out + self_message

        # 目标节点额外接收邻居消息
        if target_mask.sum() > 0:
            out[target_mask] = out[target_mask] + neighbor_message[target_mask]

        out = self.dropout(out)
        out = F.relu(out + self.bias)

        return out


class MultiLayerQueryCenteredGNN(nn.Module):
    """
    多层以"?"为中心的GNN
    按BFS距离从外向内逐层聚合
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_relations: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 第一层：input_dim -> hidden_dim
        self.first_layer = QueryCenteredBFSLayer(
            input_dim, hidden_dim, num_relations, dropout
        )

        # 中间层：hidden_dim -> hidden_dim
        self.mid_layers = nn.ModuleList([
            QueryCenteredBFSLayer(hidden_dim, hidden_dim, num_relations, dropout)
            for _ in range(num_layers - 2)
        ])

        # 最后一层：hidden_dim -> output_dim
        self.last_layer = QueryCenteredBFSLayer(
            hidden_dim, output_dim, num_relations, dropout
        )

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ] + [nn.LayerNorm(output_dim)])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        distances: torch.Tensor,
        max_distance: int
    ) -> torch.Tensor:
        """
        前向传播 - 按BFS距离从外向内聚合

        Args:
            x: 初始节点特征 (num_nodes, input_dim)
            edge_index: 边索引 (2, num_edges)
            edge_types: 边类型 (num_edges,)
            distances: 每个节点到"?"的BFS距离 (num_nodes,)
            max_distance: 最大BFS距离

        Returns:
            x: 最终的节点表征 (num_nodes, output_dim)
        """
        # 按距离从大到小处理：max_distance -> max_distance-1 -> ... -> 1 -> 0
        # 距离为0的节点就是"?"节点

        layers = [self.first_layer] + list(self.mid_layers) + [self.last_layer]

        for i, (layer, norm) in enumerate(zip(layers, self.layer_norms)):
            # 计算当前层应该处理的距离
            # 我们希望从最外层开始，所以按 max_distance - i 的顺序
            current_distance = max(0, max_distance - i)

            # 如果当前距离已经小于等于0，只更新"?"节点自身
            if current_distance <= 0:
                current_distance = 1  # 至少处理距离1到0的传递

            x = layer(x, edge_index, edge_types, distances, current_distance)
            x = norm(x)

        return x


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (GAT)
    作为R-GCN的替代方案
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.concat = concat

        # 每个头的输出维度
        self.head_dim = out_dim // num_heads if concat else out_dim

        # 关系特定的线性变换
        self.relation_linears = nn.ModuleList([
            nn.Linear(in_dim, self.head_dim * num_heads if concat else self.head_dim)
            for _ in range(num_relations)
        ])

        # 注意力参数
        self.att_src = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.randn(num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 (num_nodes, in_dim)
            edge_index: 边索引 (2, num_edges)
            edge_types: 边类型 (num_edges,)

        Returns:
            out: 更新后的节点特征
        """
        num_nodes = x.size(0)
        device = x.device

        # 聚合每种关系的输出
        outputs = []

        for r in range(self.num_relations):
            mask = edge_types == r
            if mask.sum() == 0:
                continue

            # 线性变换
            x_r = self.relation_linears[r](x)  # (num_nodes, head_dim * num_heads)

            if self.concat:
                x_r = x_r.view(num_nodes, self.num_heads, self.head_dim)  # (N, H, D)
            else:
                x_r = x_r.view(num_nodes, self.head_dim)  # (N, D)

            # 计算注意力（简化版）
            # 这里使用简单的均值聚合作为示例
            src, dst = edge_index[0][mask], edge_index[1][mask]

            # 聚合邻居信息
            out = torch.zeros_like(x_r)
            out.index_add_(0, dst, x_r[src])

            # 归一化
            degree = torch.bincount(dst, minlength=num_nodes).float().clamp(min=1)
            if self.concat:
                degree = degree.view(-1, 1, 1)
            else:
                degree = degree.view(-1, 1)
            out = out / degree

            outputs.append(out)

        # 合并不同关系的输出
        if len(outputs) == 0:
            return torch.zeros(num_nodes, self.out_dim, device=device)

        out = torch.cat(outputs, dim=-1) if self.concat else sum(outputs) / len(outputs)

        # 调整维度以匹配输出
        if out.size(-1) != self.out_dim:
            out = out.view(num_nodes, -1)[:, :self.out_dim]

        return self.dropout(out)
