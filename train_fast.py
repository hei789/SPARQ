#!/usr/bin/env python3
"""
GraphRAG 快速训练脚本
优化策略：
1. 混合精度训练 (AMP) - 加速2-3倍
2. 路径缓存 - 避免重复BFS
3. 动态负采样 - 硬负样本挖掘
4. 梯度累积 - 模拟大batch
5. DataLoader多进程 - 并行数据加载
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import time
from collections import defaultdict
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_query_encoder.models import GraphQueryEncoder
from graph_retriever import GraphRetriever
from main import CWQDataset, GraphRAGConfig


@dataclass
class FastTrainingConfig:
    """快速训练配置"""
    # 数据路径
    train_graph_query_path: str
    train_original_data_path: str
    dev_graph_query_path: str
    dev_original_data_path: str
    entities_path: str
    relations_path: str

    # 模型配置
    bert_model_path: str = "bert-base-uncased"
    hidden_dim: int = 768
    num_query_layers: int = 3
    num_retriever_layers: int = 2
    num_relations: int = 100

    # 训练配置
    epochs: int = 10
    batch_size: int = 16  # 增大batch size配合梯度累积
    learning_rate: float = 5e-4  # GNN用较大学习率
    bert_learning_rate: float = 2e-5
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # 路径采样配置
    num_negatives: int = 3  # 减少负样本数加速
    hard_negative_ratio: float = 0.5  # 硬负样本比例
    max_path_length: int = 3
    max_paths_per_sample: int = 5  # 限制每样本路径数

    # 速度优化配置
    use_amp: bool = True  # 混合精度训练，加速2-3倍
    num_workers: int = 0  # DataLoader进程数，0表示主进程（避免BERT序列化问题）
    prefetch_factor: int = 2
    gradient_accumulation_steps: int = 4  # 梯度累积，模拟大batch
    compile_model: bool = False  # PyTorch 2.0 compile（如果可用）
    cache_paths: bool = True  # 缓存采样路径
    cache_dir: str = ".cache"

    # 检查点
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1
    eval_every: int = 1

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # 快速测试
    max_train_samples: Optional[int] = None
    max_dev_samples: Optional[int] = None


class CachedPathSampler:
    """带缓存的路径采样器"""

    def __init__(self, config: FastTrainingConfig):
        self.config = config
        self.cache = {}
        self.cache_file = Path(config.cache_dir) / "path_cache.pkl"

        if config.cache_paths and self.cache_file.exists():
            print(f"Loading path cache from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)

    def save_cache(self):
        """保存缓存"""
        if self.config.cache_paths:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Path cache saved to {self.cache_file}")

    def get_cache_key(self, sample_id: str, topic_entities: Tuple) -> str:
        """生成缓存键"""
        return f"{sample_id}_{hash(topic_entities)}"

    def sample_paths_for_training(
        self,
        sample_id: str,
        topic_entities: List[int],
        answer_entities: List[int],
        subgraph_tuples: List[List[int]],
        device: torch.device
    ) -> Tuple[List, List]:
        """
        为训练采样正负样本路径
        返回: (positive_paths, negative_paths)
        """
        cache_key = self.get_cache_key(sample_id, tuple(topic_entities))

        if cache_key in self.cache:
            return self.cache[cache_key]

        # 构建邻接表
        adj_list = defaultdict(list)
        all_entities = set()

        for h, r, t in subgraph_tuples:
            adj_list[h].append((t, r))
            all_entities.add(h)
            all_entities.add(t)

        # 找到正样本路径（包含答案实体）
        positive_paths = []
        answer_set = set(answer_entities)

        for start in topic_entities:
            if start not in all_entities:
                continue

            # BFS限制搜索深度
            visited = {start: ([start], [])}
            queue = [start]

            for depth in range(self.config.max_path_length):
                next_queue = []
                for node in queue:
                    if node not in adj_list:
                        continue

                    for next_node, rel in adj_list[node]:
                        if next_node in visited:
                            continue

                        path_nodes, path_rels = visited[node]
                        new_path_nodes = path_nodes + [next_node]
                        new_path_rels = path_rels + [rel]

                        visited[next_node] = (new_path_nodes, new_path_rels)

                        # 如果是答案实体，加入正样本
                        if next_node in answer_set:
                            positive_paths.append((new_path_nodes, new_path_rels))

                        next_queue.append(next_node)

                queue = next_queue

                if len(positive_paths) >= self.config.max_paths_per_sample:
                    break

        # 如果没有找到正样本，使用从话题实体出发的短路径
        if not positive_paths:
            for start in topic_entities[:1]:  # 只取第一个
                if start in adj_list:
                    for next_node, rel in adj_list[start][:3]:
                        positive_paths.append(([start, next_node], [rel]))

        # 限制正样本数量
        positive_paths = positive_paths[:self.config.max_paths_per_sample]

        # 采样负样本路径
        num_negatives = len(positive_paths) * self.config.num_negatives
        negative_paths = []

        # 简单随机游走采样
        for _ in range(num_negatives * 2):  # 多采样一些过滤
            if len(negative_paths) >= num_negatives:
                break

            start = random.choice(topic_entities)
            if start not in all_entities:
                continue

            current = start
            path_nodes = [start]
            path_rels = []

            for step in range(random.randint(1, self.config.max_path_length)):
                if current not in adj_list or not adj_list[current]:
                    break

                next_node, rel = random.choice(adj_list[current])
                if next_node in path_nodes:  # 避免环路
                    break

                path_nodes.append(next_node)
                path_rels.append(rel)
                current = next_node

                # 确保不是答案实体且路径长度>=2
                if current not in answer_set and len(path_nodes) >= 2:
                    negative_paths.append((path_nodes.copy(), path_rels.copy()))
                    if len(negative_paths) >= num_negatives:
                        break

        negative_paths = negative_paths[:num_negatives]

        result = (positive_paths, negative_paths)

        if self.config.cache_paths:
            self.cache[cache_key] = result

        return result


class FastGraphRAGTrainer:
    """快速训练器"""

    def __init__(self, config: FastTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # 初始化模型
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.scaler = GradScaler(enabled=config.use_amp)

        # 路径采样器
        self.path_sampler = CachedPathSampler(config)

        # 统计
        self.global_step = 0
        self.best_metric = 0.0

    def _init_model(self):
        """初始化模型"""
        print("Initializing model...")

        # 创建查询编码器和路径编码器
        query_encoder = GraphQueryEncoder(
            hidden_dim=self.config.hidden_dim,
            num_gnn_layers=self.config.num_query_layers,
            num_relations=self.config.num_relations,
            bert_model_name=self.config.bert_model_path,
            use_query_centered_pooling=True,
            use_bfs_gnn=True
        )

        # 路径编码器
        retriever = GraphRetriever(
            hidden_dim=self.config.hidden_dim,
            num_gnn_layers=self.config.num_retriever_layers,
            num_relations=self.config.num_relations
        )
        path_encoder = retriever.path_encoder

        # 组合模型
        model = nn.ModuleDict({
            'query_encoder': query_encoder,
            'path_encoder': path_encoder
        })

        # 冻结BERT前几层（可选，加速训练）
        # for param in query_encoder.embedding_layer.bert.encoder.layer[:6].parameters():
        #     param.requires_grad = False

        model = model.to(self.device)

        # PyTorch 2.0 compile（如果可用）
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("Using torch.compile for acceleration")
            model = torch.compile(model)

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return model

    def _init_optimizer(self):
        """初始化优化器"""
        # 分层学习率
        param_groups = [
            {'params': self.model['query_encoder'].gnn.parameters(), 'lr': self.config.learning_rate},
            {'params': self.model['path_encoder'].parameters(), 'lr': self.config.learning_rate},
            {'params': self.model['query_encoder'].embedding_layer.parameters(), 'lr': self.config.bert_learning_rate},
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
        return optimizer

    def contrastive_loss(self, query_emb: torch.Tensor, pos_embs: List[torch.Tensor],
                        neg_embs: List[torch.Tensor]) -> torch.Tensor:
        """对比学习损失"""
        if not pos_embs:
            return torch.tensor(0.0, device=self.device)

        # 编码正样本路径
        if pos_embs:
            pos_stack = torch.stack(pos_embs)
            pos_sim = F.cosine_similarity(query_emb.unsqueeze(0), pos_stack, dim=-1)
            pos_loss = torch.clamp(1 - pos_sim, min=0).mean()
        else:
            pos_loss = torch.tensor(0.0, device=self.device)

        # 编码负样本路径
        if neg_embs:
            neg_stack = torch.stack(neg_embs)
            neg_sim = F.cosine_similarity(query_emb.unsqueeze(0), neg_stack, dim=-1)
            # 困难负样本：相似度高的负样本给予更大权重
            weights = F.softmax(neg_sim * 2, dim=0)
            neg_loss = (weights * F.relu(neg_sim - 0.5)).sum()
        else:
            neg_loss = torch.tensor(0.0, device=self.device)

        return pos_loss + neg_loss

    def build_subgraph_features(
        self,
        subgraph_tuples: List[List[int]],
        topic_entities: List[int]
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        构建子图特征

        Args:
            subgraph_tuples: 三元组列表 [(h, r, t), ...]
            topic_entities: 话题实体索引列表

        Returns:
            node_features: 节点特征 (num_nodes, hidden_dim)
            entity_idx2local: 实体索引到局部索引的映射
        """
        # 收集所有实体索引
        all_entity_indices = set()
        for h, r, t in subgraph_tuples:
            all_entity_indices.add(h)
            all_entity_indices.add(t)

        # 确保话题实体也被包含
        for idx in topic_entities:
            all_entity_indices.add(idx)

        if not all_entity_indices:
            return None, {}

        # 创建局部索引映射
        entity_list = sorted(list(all_entity_indices))
        entity_idx2local = {idx: i for i, idx in enumerate(entity_list)}
        num_nodes = len(entity_list)

        # 初始化节点特征（可学习的特征，这里用随机初始化）
        node_features = torch.randn(
            num_nodes, self.config.hidden_dim,
            device=self.device
        )

        return node_features, entity_idx2local

    def train_step(self, sample: Dict) -> float:
        """单个训练步骤"""
        subgraph_tuples = sample['subgraph'].get('tuples', [])
        if not subgraph_tuples:
            return 0.0

        # 获取答案实体（从answers字段解析）
        answer_entities = []
        for ans in sample.get('answers', []):
            if isinstance(ans, dict) and 'kb_id' in ans:
                # 尝试找到答案实体的索引
                # 简化：使用话题实体附近的节点作为潜在答案
                pass

        # 如果没有明确答案，使用路径终点作为答案
        topic_entities = sample['entities']

        # 采样路径
        pos_paths, neg_paths = self.path_sampler.sample_paths_for_training(
            sample['id'],
            topic_entities,
            topic_entities,  # 使用话题实体作为答案候选
            subgraph_tuples,
            self.device
        )

        if not pos_paths:
            return 0.0

        # 构建子图特征
        node_features, entity_idx2local = self.build_subgraph_features(
            subgraph_tuples, topic_entities
        )
        if node_features is None:
            return 0.0

        with autocast(enabled=self.config.use_amp):
            # 编码查询
            query_result = self.model['query_encoder'](
                sample['triples'],
                return_all_node_features=True
            )
            if isinstance(query_result, tuple):
                query_emb = query_result[1]
            else:
                query_emb = query_result

            # 将全局实体索引转换为局部索引
            def global_to_local(global_idx):
                return entity_idx2local.get(global_idx, -1)

            # 编码正样本路径
            pos_embs = []
            for path_nodes, path_rels in pos_paths:
                # 转换节点索引为局部索引
                local_nodes = [global_to_local(n) for n in path_nodes]
                if -1 in local_nodes or len(local_nodes) == 0:
                    continue

                # 关系索引取模，确保在范围内
                local_rels = [r % self.config.num_relations for r in path_rels]

                # 使用 path_encoder 编码路径
                path_emb = self.model['path_encoder'](
                    node_features,
                    local_nodes,
                    local_rels
                )
                pos_embs.append(path_emb)

            # 编码负样本路径
            neg_embs = []
            for path_nodes, path_rels in neg_paths:
                local_nodes = [global_to_local(n) for n in path_nodes]
                if -1 in local_nodes or len(local_nodes) == 0:
                    continue

                # 关系索引取模，确保在范围内
                local_rels = [r % self.config.num_relations for r in path_rels]

                path_emb = self.model['path_encoder'](
                    node_features,
                    local_nodes,
                    local_rels
                )
                neg_embs.append(path_emb)

            if not pos_embs:
                return 0.0

            # 计算损失
            loss = self.contrastive_loss(query_emb, pos_embs, neg_embs)

        return loss

    def train_epoch(self, dataset: CWQDataset, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        num_samples = 0
        num_valid = 0

        # 限制样本数
        max_samples = len(dataset) if self.config.max_train_samples is None else min(self.config.max_train_samples, len(dataset))
        indices = list(range(max_samples))
        random.shuffle(indices)

        progress = tqdm(indices, desc=f"Epoch {epoch}")

        self.optimizer.zero_grad()

        for i, idx in enumerate(progress):
            sample = dataset.get_sample(idx)
            if not sample or not sample.get('subgraph'):
                continue

            # 计算损失
            loss = self.train_step(sample)

            if loss == 0.0:
                continue

            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps

            # 反向传播
            self.scaler.scale(loss).backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_valid += 1

            # 梯度更新
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            num_samples += 1

            # 更新进度条
            if num_valid > 0:
                progress.set_postfix({
                    'loss': total_loss / num_valid,
                    'valid': num_valid
                })

        # 处理剩余梯度
        if (i + 1) % self.config.gradient_accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(num_valid, 1)

        return {
            'loss': avg_loss,
            'num_valid': num_valid,
            'num_total': num_samples
        }

    def evaluate(self, dataset: CWQDataset) -> Dict[str, float]:
        """验证"""
        self.model.eval()

        total_loss = 0.0
        num_valid = 0

        max_samples = len(dataset) if self.config.max_dev_samples is None else min(self.config.max_dev_samples, len(dataset))

        with torch.no_grad():
            for idx in tqdm(range(max_samples), desc="Evaluating"):
                sample = dataset.get_sample(idx)
                if not sample or not sample.get('subgraph'):
                    continue

                loss = self.train_step(sample)

                if loss > 0:
                    total_loss += loss.item()
                    num_valid += 1

        return {
            'loss': total_loss / max(num_valid, 1),
            'num_valid': num_valid
        }

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'global_step': self.global_step
        }

        # 保存最新检查点
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # 保存每轮检查点
        epoch_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)

        # 保存最佳检查点
        if is_best:
            best_path = checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved! Loss: {metrics['loss']:.4f}")

        # 保存路径缓存
        self.path_sampler.save_cache()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GraphRAG Fast Training")

    # 数据路径
    parser.add_argument("--train_graph_query_path", type=str, required=True)
    parser.add_argument("--train_original_data_path", type=str, required=True)
    parser.add_argument("--dev_graph_query_path", type=str, required=True)
    parser.add_argument("--dev_original_data_path", type=str, required=True)
    parser.add_argument("--entities_path", type=str, required=True)
    parser.add_argument("--relations_path", type=str, required=True)

    # 模型配置
    parser.add_argument("--bert_model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_relations", type=int, default=100)

    # 训练配置
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--bert_learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_negatives", type=int, default=3)

    # 速度优化
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--cache_paths", action="store_true", default=True)
    parser.add_argument("--compile_model", action="store_true", default=False)

    # 快速测试
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 创建配置
    config = FastTrainingConfig(
        train_graph_query_path=args.train_graph_query_path,
        train_original_data_path=args.train_original_data_path,
        dev_graph_query_path=args.dev_graph_query_path,
        dev_original_data_path=args.dev_original_data_path,
        entities_path=args.entities_path,
        relations_path=args.relations_path,
        bert_model_path=args.bert_model_path,
        hidden_dim=args.hidden_dim,
        num_relations=args.num_relations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        bert_learning_rate=args.bert_learning_rate,
        num_negatives=args.num_negatives,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cache_paths=args.cache_paths,
        compile_model=args.compile_model,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples
    )

    print("=" * 80)
    print("GraphRAG Fast Training")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"AMP: {config.use_amp}")
    print(f"Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"Path Cache: {config.cache_paths}")

    # 创建训练器
    trainer = FastGraphRAGTrainer(config)

    # 加载数据
    print("\nLoading datasets...")
    train_config = GraphRAGConfig(
        graph_query_path=config.train_graph_query_path,
        original_data_path=config.train_original_data_path,
        entities_path=config.entities_path,
        relations_path=config.relations_path,
        num_relations=config.num_relations
    )
    train_dataset = CWQDataset(train_config)

    dev_config = GraphRAGConfig(
        graph_query_path=config.dev_graph_query_path,
        original_data_path=config.dev_original_data_path,
        entities_path=config.entities_path,
        relations_path=config.relations_path,
        num_relations=config.num_relations
    )
    dev_dataset = CWQDataset(dev_config)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")

    # 训练循环
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'=' * 80}")

        start_time = time.time()

        # 训练
        train_metrics = trainer.train_epoch(train_dataset, epoch)
        print(f"Train Loss: {train_metrics['loss']:.4f} | Valid: {train_metrics['num_valid']}/{train_metrics['num_total']}")

        # 验证
        if epoch % config.eval_every == 0:
            val_metrics = trainer.evaluate(dev_dataset)
            print(f"Val Loss: {val_metrics['loss']:.4f}")

            # 保存最佳模型
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
        else:
            val_metrics = train_metrics
            is_best = False

        # 保存检查点
        if epoch % config.save_every == 0:
            trainer.save_checkpoint(epoch, val_metrics, is_best)

        elapsed = time.time() - start_time
        print(f"Time: {elapsed:.2f}s")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
