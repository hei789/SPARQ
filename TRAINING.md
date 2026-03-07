# GraphRAG 训练方案

## 目标

训练图神经网络检索器，使其能够从话题实体出发，通过束搜索找到包含**答案实体**的推理路径。

## 核心思想

- **正样本**：包含答案实体的推理路径
- **负样本**：不包含答案实体的推理路径
- **训练目标**：让查询表征与正样本路径的相似度 > 与负样本路径的相似度

## 数据准备

### 正样本构建

对于每个训练样本：

1. 从话题实体出发，在子图中进行BFS遍历
2. 收集所有能够到达答案实体的路径
3. 这些路径作为正样本

```python
def find_answer_paths(
    topic_entities: List[int],
    answer_entities: List[int],
    subgraph: Dict,
    max_length: int = 3
) -> List[ReasoningPath]:
    """找到从话题实体到答案实体的所有路径"""
    answer_paths = []
    for start in topic_entities:
        for end in answer_entities:
            # BFS寻找最短路径
            paths = bfs_all_paths(start, end, subgraph, max_length)
            for path in paths:
                answer_paths.append(path)
    return answer_paths
```

### 负样本采样

对于每个正样本路径，采样 `k` 个负样本：

1. 从同一话题实体出发
2. 在子图中随机游走
3. 确保路径终点不是答案实体

## 训练目标

### 1. 对比学习损失（Contrastive Loss）

让查询与正样本的相似度远大于与负样本的相似度：

```python
def contrastive_loss(
    query_embedding: torch.Tensor,  # (hidden_dim,)
    positive_paths: torch.Tensor,    # (num_pos, hidden_dim)
    negative_paths: torch.Tensor,    # (num_neg, hidden_dim)
    margin: float = 0.5
) -> torch.Tensor:
    """对比学习损失"""
    # 计算相似度（余弦相似度）
    pos_sim = F.cosine_similarity(query_embedding.unsqueeze(0), positive_paths, dim=-1)
    neg_sim = F.cosine_similarity(query_embedding.unsqueeze(0), negative_paths, dim=-1)

    # 正样本损失：希望相似度接近1
    pos_loss = (1 - pos_sim).mean()

    # 负样本损失：希望相似度接近0
    neg_loss = F.relu(neg_sim - margin).mean()

    return pos_loss + neg_loss
```

### 2. 多任务损失

同时优化路径检索和答案预测：

```python
def combined_loss(
    query_embedding: torch.Tensor,
    path_embeddings: torch.Tensor,      # (num_paths, hidden_dim)
    path_scores: torch.Tensor,          # (num_paths,)
    answer_entity_embedding: torch.Tensor,
    answer_logits: torch.Tensor,        # (num_entities,)
    answer_labels: torch.Tensor,        # (num_entities,) 0/1
) -> torch.Tensor:
    """组合损失"""
    # 1. 对比损失
    contrast_loss = contrastive_loss(...)

    # 2. 路径排序损失（RankNet）
    # 让正样本路径的分数 > 负样本路径的分数
    pos_mask = ...  # 正样本掩码
    neg_mask = ...  # 负样本掩码
    pos_scores = path_scores[pos_mask]
    neg_scores = path_scores[neg_mask]
    rank_loss = F.margin_ranking_loss(
        pos_scores, neg_scores,
        target=torch.ones(len(pos_scores)),
        margin=0.5
    )

    # 3. 答案实体分类损失
    answer_loss = F.binary_cross_entropy_with_logits(answer_logits, answer_labels)

    return contrast_loss + rank_loss + answer_loss
```

## 模型架构调整

### 1. 路径编码器改进

在 `PathEncoder` 中添加答案实体注意力：

```python
class PathEncoder(nn.Module):
    def forward(
        self,
        node_features: torch.Tensor,
        path_nodes: List[int],
        path_relations: List[int],
        answer_entity_idx: Optional[int] = None  # 新增
    ) -> torch.Tensor:
        # ... 原有编码逻辑 ...

        # 如果知道答案实体位置，添加注意力权重
        if answer_entity_idx is not None and answer_entity_idx in path_nodes:
            # 增加答案实体在路径中的权重
            attention_weights = torch.ones(len(path_nodes))
            answer_pos = path_nodes.index(answer_entity_idx)
            attention_weights[answer_pos] *= 2.0
            # ...
```

### 2. 答案预测头

在检索器后添加答案预测模块：

```python
class GraphRetrieverWithAnswer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 原有检索器组件
        self.retriever = GraphRetriever(...)

        # 新增答案预测头
        self.answer_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, ...):
        # 1. 检索路径
        paths = self.retriever(...)

        # 2. 预测每个实体是答案的概率
        answer_logits = []
        for i in range(num_nodes):
            # 结合节点特征和查询特征
            combined = torch.cat([encoded_features[i], query_embedding], dim=-1)
            logit = self.answer_predictor(combined)
            answer_logits.append(logit)

        return paths, torch.cat(answer_logits)
```

## 训练流程

### 步骤1：数据准备

```python
def prepare_training_data(dataset, num_negatives=5):
    """准备训练数据"""
    training_samples = []

    for sample in dataset:
        # 1. 找到所有答案实体
        answer_entities = [e['kb_id'] for e in sample['answers']]

        # 2. 找到正样本路径（包含答案的路径）
        positive_paths = find_answer_paths(
            topic_entities=sample['entities'],
            answer_entities=answer_entities,
            subgraph=sample['subgraph']
        )

        # 3. 采样负样本路径（不包含答案的路径）
        negative_paths = sample_negative_paths(
            topic_entities=sample['entities'],
            answer_entities=answer_entities,
            subgraph=sample['subgraph'],
            num_samples=len(positive_paths) * num_negatives
        )

        training_samples.append({
            'question': sample['question'],
            'triples': sample['triples'],
            'subgraph': sample['subgraph'],
            'positive_paths': positive_paths,
            'negative_paths': negative_paths,
            'answer_entities': answer_entities
        })

    return training_samples
```

### 步骤2：训练循环

```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        # 1. 编码查询
        query_embedding = model.query_encoder(batch['triples'])

        # 2. 编码所有候选路径
        path_embeddings = []
        path_labels = []  # 1=正样本, 0=负样本

        for path in batch['positive_paths']:
            emb = model.path_encoder(path)
            path_embeddings.append(emb)
            path_labels.append(1)

        for path in batch['negative_paths']:
            emb = model.path_encoder(path)
            path_embeddings.append(emb)
            path_labels.append(0)

        path_embeddings = torch.stack(path_embeddings)  # (num_paths, hidden_dim)
        path_labels = torch.tensor(path_labels, device=device)

        # 3. 计算相似度
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            path_embeddings,
            dim=-1
        )

        # 4. 计算损失
        loss = contrastive_loss(
            query_embedding,
            path_embeddings[path_labels == 1],
            path_embeddings[path_labels == 0]
        )

        # 5. 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

## 优化器和学习率

```python
# 优化器选择
optimizer = torch.optim.Adam([
    {'params': model.query_encoder.parameters(), 'lr': 1e-5},  # BERT用较小学习率
    {'params': model.retriever.parameters(), 'lr': 1e-3},       # GNN用较大学习率
], weight_decay=1e-5)

# 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
```

## 评估指标

### 1. 路径检索准确率

```python
def evaluate_path_retrieval(model, dataloader, top_k=10):
    """评估路径检索性能"""
    hits_at_k = 0
    total = 0

    for batch in dataloader:
        # 检索路径
        retrieved_paths = model.retrieve(batch, top_k=top_k)

        # 检查是否有路径包含答案
        for i, paths in enumerate(retrieved_paths):
            answer_entities = batch['answer_entities'][i]
            has_answer = any(
                path_contains_answer(p, answer_entities)
                for p in paths
            )
            if has_answer:
                hits_at_k += 1
            total += 1

    return hits_at_k / total
```

### 2. 答案实体准确率

```python
def evaluate_answer_prediction(model, dataloader):
    """评估答案预测性能"""
    correct = 0
    total = 0

    for batch in dataloader:
        _, answer_logits = model(batch)

        # Top-1准确率
        predicted = answer_logits.argmax(dim=-1)
        correct += (predicted == batch['answer_labels']).sum().item()
        total += len(batch)

    return correct / total
```

## 训练技巧

### 1. 硬负样本挖掘

```python
def hard_negative_mining(model, query_embedding, candidates, num_hard=3):
    """选择相似度高的负样本作为硬负样本"""
    similarities = F.cosine_similarity(
        query_embedding.unsqueeze(0),
        candidates,
        dim=-1
    )
    # 选择相似度最高的负样本
    _, indices = similarities.topk(num_hard)
    return candidates[indices]
```

### 2. 渐进式训练

```python
# Stage 1: 固定BERT，只训练GNN
for param in model.query_encoder.embedding_layer.parameters():
    param.requires_grad = False

# Stage 2: 解冻BERT，联合训练
for param in model.query_encoder.embedding_layer.parameters():
    param.requires_grad = True
```

### 3. 数据增强

```python
def augment_subgraph(subgraph, drop_rate=0.1):
    """随机删除部分边进行数据增强"""
    tuples = subgraph['tuples']
    num_drop = int(len(tuples) * drop_rate)
    keep_indices = random.sample(range(len(tuples)), len(tuples) - num_drop)
    return {'tuples': [tuples[i] for i in keep_indices]}
```

## 实现建议

1. **先实现基础版本**：只用对比损失训练路径检索
2. **逐步添加功能**：添加答案预测头、硬负样本挖掘等
3. **调试技巧**：先用少量数据验证损失下降，再全量训练
4. **保存检查点**：每轮保存，选择验证集上Hits@10最高的模型

## 快速开始

### 1. 基础训练

```bash
python SPARQ/train.py \
    --train_graph_query_path dataset/CWQ/CWQ/graph_query_train.json \
    --train_original_data_path dataset/CWQ/CWQ/train_simple.json \
    --dev_graph_query_path dataset/CWQ/CWQ/graph_query_dev.json \
    --dev_original_data_path dataset/CWQ/CWQ/dev_simple.json \
    --entities_path dataset/CWQ/CWQ/entities.txt \
    --relations_path dataset/CWQ/CWQ/relations.txt \
    --bert_model_path /path/to/bert-base-uncased \
    --epochs 10 \
    --batch_size 8
```

### 2. 快速训练（推荐）

使用 `train_fast.py` 进行优化训练，速度提升 **2-5倍**：

```bash
python SPARQ/train_fast.py \
    --train_graph_query_path dataset/CWQ/CWQ/graph_query_train.json \
    --train_original_data_path dataset/CWQ/CWQ/train_simple.json \
    --dev_graph_query_path dataset/CWQ/CWQ/graph_query_dev.json \
    --dev_original_data_path dataset/CWQ/CWQ/dev_simple.json \
    --entities_path dataset/CWQ/CWQ/entities.txt \
    --relations_path dataset/CWQ/CWQ/relations.txt \
    --bert_model_path /path/to/bert-base-uncased \
    --epochs 10 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --use_amp \
    --cache_paths
```

---

## train_fast.py 详细训练方法

### 1. 训练架构

`train_fast.py` 采用**双塔对比学习架构**，训练时冻结路径编码器（path_encoder），仅训练查询编码器（query_encoder）：

```
┌─────────────────┐     ┌─────────────────┐
│  Query Encoder  │     │  Path Encoder   │
│  (trainable)    │     │  (frozen)       │
│                 │     │                 │
│  BERT + GNN     │     │  GNN Layers     │
│  ↓              │     │  ↓              │
│  query_emb      │◄────┤  path_emb       │
│  (768-dim)      │相似度 │  (768-dim)      │
└─────────────────┘     └─────────────────┘
         │                       │
         │         对比学习       │
         └───────┬───────────────┘
                 ▼
           contrastive_loss
```

### 2. 正负样本路径采样

#### 2.1 正样本路径采样

正样本路径定义为：**从话题实体出发，通过BFS遍历能够到达答案实体的路径**。

采样算法：
```python
def sample_positive_paths(topic_entities, answer_entities, subgraph, max_length=3):
    positive_paths = []
    for start in topic_entities:
        # BFS遍历，深度限制为 max_length
        for depth in range(max_path_length):
            for node in current_queue:
                for (next_node, relation) in adj_list[node]:
                    if next_node in answer_set:
                        # 找到答案实体，记录路径
                        positive_paths.append((path_nodes, path_relations))
    return positive_paths[:max_paths_per_sample]  # 限制每样本路径数
```

**参数说明：**
| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| `max_path_length` | $L_{max}$ | 3 | 最大路径长度（跳数） |
| `max_paths_per_sample` | $K_{pos}$ | 5 | 每样本最大正样本路径数 |

#### 2.2 负样本路径采样

负样本路径通过**随机游走**生成，确保终点不是答案实体：

```python
def sample_negative_paths(topic_entities, answer_entities, subgraph, num_negatives):
    negative_paths = []
    for _ in range(num_negatives):
        start = random.choice(topic_entities)
        path_nodes = [start]
        path_rels = []

        for step in range(random.randint(1, max_path_length)):
            next_node, rel = random.choice(adj_list[current])
            if next_node in path_nodes:  # 避免环路
                break
            path_nodes.append(next_node)
            path_rels.append(rel)

            # 确保不是答案实体且路径长度>=2
            if current not in answer_set and len(path_nodes) >= 2:
                negative_paths.append((path_nodes, path_rels))
```

**参数说明：**
| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| `num_negatives` | $N_{neg}$ | 3 | 每个正样本对应的负样本数量 |

### 3. 损失函数计算

#### 3.1 余弦相似度计算

查询表征与路径表征之间的相似度使用**余弦相似度**计算：

$$\text{sim}(q, p) = \frac{q \cdot p}{\|q\|_2 \|p\|_2} = \frac{\sum_{i=1}^{d} q_i p_i}{\sqrt{\sum_{i=1}^{d} q_i^2} \cdot \sqrt{\sum_{i=1}^{d} p_i^2}}$$

其中：
- $q \in \mathbb{R}^{d}$：查询表征向量，$d = 768$
- $p \in \mathbb{R}^{d}$：路径表征向量，$d = 768$
- $\text{sim}(q, p) \in [-1, 1]$：相似度值

**数值稳定版本**（防止除零）：
```python
def safe_cosine_similarity(x1, x2, eps=1e-8):
    x1_norm = F.normalize(x1, p=2, dim=-1, eps=eps)
    x2_norm = F.normalize(x2, p=2, dim=-1, eps=eps)
    return (x1_norm * x2_norm).sum(dim=-1).clamp(-1 + eps, 1 - eps)
```

#### 3.2 对比学习损失（Contrastive Loss）

总损失由**正样本损失**和**负样本损失**两部分组成：

$$\mathcal{L} = \mathcal{L}_{pos} + \mathcal{L}_{neg}$$

##### 正样本损失

对于每个正样本路径 $p^+ \in \mathcal{P}^+$，希望其与查询的相似度接近1：

$$\mathcal{L}_{pos} = \frac{1}{|\mathcal{P}^+|} \sum_{p^+ \in \mathcal{P}^+} \max(0, 1 - \text{sim}(q, p^+))$$

其中：
- $\mathcal{P}^+$：正样本路径集合
- $|\mathcal{P}^+|$：正样本路径数量
- $\max(0, 1 - \text{sim}(q, p^+))$：Hinge Loss，当相似度<1时产生惩罚

##### 负样本损失

对于负样本路径 $p^- \in \mathcal{P}^-$，采用**温度缩放的加权损失**：

$$\mathcal{L}_{neg} = \sum_{p^- \in \mathcal{P}^-} w(p^-) \cdot \max(0, \text{sim}(q, p^-) - \tau)$$

其中权重 $w(p^-)$ 通过softmax计算：

$$w(p^-) = \frac{\exp(\text{sim}(q, p^-) / T)}{\sum_{p' \in \mathcal{P}^-} \exp(\text{sim}(q, p') / T)}$$

**参数说明：**
| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| `margin` | $\tau$ | 0.3 | 负样本边界阈值，相似度低于此值不惩罚 |
| `temperature` | $T$ | 0.1 | 温度系数，控制softmax分布的平滑程度 |

#### 3.3 完整损失函数代码

```python
def contrastive_loss(query_emb, pos_embs, neg_embs):
    """
    参数:
        query_emb: (hidden_dim,) - 查询表征
        pos_embs: List[(hidden_dim,)] - 正样本路径表征列表
        neg_embs: List[(hidden_dim,)] - 负样本路径表征列表
    返回:
        loss: scalar - 对比学习损失值
    """
    eps = 1e-8

    # 正样本损失
    pos_stack = torch.stack(pos_embs)  # (num_pos, hidden_dim)
    pos_sim = safe_cosine_similarity(query_emb.unsqueeze(0), pos_stack, eps)
    pos_loss = torch.clamp(1 - pos_sim, min=0).mean()

    # 负样本损失（加权）
    if neg_embs:
        neg_stack = torch.stack(neg_embs)  # (num_neg, hidden_dim)
        neg_sim = safe_cosine_similarity(query_emb.unsqueeze(0), neg_stack, eps)
        weights = F.softmax(neg_sim / 0.1, dim=0)  # 温度缩放
        neg_loss = (weights * F.relu(neg_sim - 0.3)).sum()
    else:
        neg_loss = 0

    loss = pos_loss + neg_loss

    # NaN检查
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss
```

### 4. 优化器配置

采用**分层学习率**策略：

```python
optimizer = torch.optim.AdamW([
    # GNN层使用较大学习率
    {'params': query_encoder.gnn.parameters(), 'lr': 5e-4},
    # BERT层使用较小学习率
    {'params': query_encoder.embedding_layer.parameters(), 'lr': 2e-5},
], weight_decay=1e-5)
```

**参数说明：**
| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| `learning_rate` | $\eta_{gnn}$ | $5 \times 10^{-4}$ | GNN层学习率 |
| `bert_learning_rate` | $\eta_{bert}$ | $2 \times 10^{-5}$ | BERT层学习率 |
| `weight_decay` | $\lambda$ | $10^{-5}$ | 权重衰减系数 |
| `max_grad_norm` | - | 1.0 | 梯度裁剪阈值 |

### 5. 梯度累积

为了模拟更大的batch size，使用梯度累积：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N_{acc}} \sum_{i=1}^{N_{acc}} \nabla_\theta \mathcal{L}_i$$

其中 $N_{acc}$ = `gradient_accumulation_steps`（默认4）。

实际batch size = `batch_size` × `gradient_accumulation_steps` = 16 × 4 = 64

### 6. 混合精度训练（AMP）

使用自动混合精度加速训练：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(enabled=config.use_amp)

with autocast(enabled=config.use_amp):
    loss = compute_loss(...)  # 前向传播使用FP16

scaler.scale(loss).backward()  # 反向传播
scaler.step(optimizer)
scaler.update()
```

### 7. 训练配置参数汇总

#### FastTrainingConfig 完整参数表

| 类别 | 参数名 | 类型 | 默认值 | 说明 |
|------|--------|------|--------|------|
| **数据路径** | `train_graph_query_path` | str | 必填 | 训练集图查询路径 |
| | `train_original_data_path` | str | 必填 | 训练集原始数据路径 |
| | `dev_graph_query_path` | str | 必填 | 验证集图查询路径 |
| | `dev_original_data_path` | str | 必填 | 验证集原始数据路径 |
| | `entities_path` | str | 必填 | 实体列表文件路径 |
| | `relations_path` | str | 必填 | 关系列表文件路径 |
| **模型配置** | `bert_model_path` | str | "bert-base-uncased" | 预训练BERT模型路径 |
| | `hidden_dim` | int | 768 | 隐藏层维度 |
| | `num_query_layers` | int | 3 | 查询编码器GNN层数 |
| | `num_retriever_layers` | int | 2 | 检索器GNN层数 |
| | `num_relations` | int | 100 | 关系数量 |
| **训练配置** | `epochs` | int | 10 | 训练轮数 |
| | `batch_size` | int | 16 | 批次大小 |
| | `learning_rate` | float | $5 \times 10^{-4}$ | GNN层学习率 |
| | `bert_learning_rate` | float | $2 \times 10^{-5}$ | BERT层学习率 |
| | `weight_decay` | float | $10^{-5}$ | 权重衰减 |
| | `max_grad_norm` | float | 1.0 | 梯度裁剪阈值 |
| | `warmup_steps` | int | 100 | 预热步数 |
| **路径采样** | `num_negatives` | int | 3 | 负样本数/正样本 |
| | `hard_negative_ratio` | float | 0.5 | 硬负样本比例 |
| | `max_path_length` | int | 3 | 最大路径长度 |
| | `max_paths_per_sample` | int | 5 | 每样本最大路径数 |
| **速度优化** | `use_amp` | bool | True | 混合精度训练 |
| | `num_workers` | int | 0 | DataLoader进程数 |
| | `gradient_accumulation_steps` | int | 4 | 梯度累积步数 |
| | `cache_paths` | bool | True | 是否缓存路径 |
| | `cache_dir` | str | ".cache" | 缓存目录 |
| | `compile_model` | bool | False | PyTorch 2.0编译 |
| **检查点** | `checkpoint_dir` | str | "checkpoints" | 检查点保存目录 |
| | `save_every` | int | 1 | 每N轮保存检查点 |
| | `eval_every` | int | 1 | 每N轮验证一次 |
| **其他** | `device` | str | "cuda"/"cpu" | 训练设备 |
| | `seed` | int | 42 | 随机种子 |
| | `max_train_samples` | int | None | 最大训练样本数（测试用） |
| | `max_dev_samples` | int | None | 最大验证样本数（测试用） |

## 训练速度优化

### 优化策略对比

| 优化技术 | 加速效果 | 实现方式 |
|---------|---------|---------|
| **混合精度训练 (AMP)** | 2-3x | `torch.cuda.amp` |
| **路径缓存** | 1.5-2x | 预采样并缓存正负样本路径 |
| **梯度累积** | - | 模拟大batch，减少通信开销 |
| **BERT部分冻结** | 1.2-1.5x | 冻结前6层只训练上层 |
| **减少负样本数** | 1.5x | 从5个减少到3个 |
| **路径长度限制** | 1.2x | 限制max_path_length=3 |

### 快速训练配置

```python
# train_fast.py 中的优化配置
config = FastTrainingConfig(
    use_amp=True,                      # 混合精度训练
    gradient_accumulation_steps=4,     # 梯度累积
    cache_paths=True,                  # 缓存采样路径
    num_negatives=3,                   # 减少负样本
    max_paths_per_sample=5,            # 限制路径数
    num_workers=0,                     # 避免BERT序列化问题
)
```

### 不同硬件配置建议

**单卡 16GB（如V100）**:
```bash
--batch_size 8 \
--gradient_accumulation_steps 4 \
--use_amp
```

**单卡 24GB（如RTX 3090）**:
```bash
--batch_size 16 \
--gradient_accumulation_steps 2 \
--use_amp
```

**多卡训练**:
```bash
torchrun --nproc_per_node=2 SPARQ/train_fast.py ...
```

## 参考实现位置

```
SPARQ/
├── train.py              # 基础训练入口
├── train_fast.py         # 快速训练（优化版本）
├── evaluate_test.py      # 测试集评估（计算Hit@K）
├── TRAINING.md           # 本文件
└── checkpoints/          # 检查点保存目录
    ├── best.pt           # 最佳模型
    ├── latest.pt         # 最新模型
    └── epoch_*.pt        # 每轮模型
```

---

## 测试集评估

训练完成后，使用 `evaluate_test.py` 在测试集上评估模型性能，计算 **Hit@1 ~ Hit@10** 指标。

### Hit@K 指标定义

Hit@K 表示在前 K 条检索路径中至少包含一个正确答案的样本比例：

$$\text{Hit@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left(\exists p \in \text{TopK}(q_i), \text{Answer}(q_i) \cap p \neq \emptyset\right) \times 100\%$$

其中：
- $N$：测试样本总数
- $\text{TopK}(q_i)$：查询 $q_i$ 的前 K 条检索路径
- $\text{Answer}(q_i)$：查询 $q_i$ 的答案实体集合
- $\mathbb{1}(\cdot)$：指示函数，条件满足时为1，否则为0

### 使用方法

#### 方式1：使用 shell 脚本

```bash
chmod +x eval_test.sh
./eval_test.sh checkpoints/best.pt
```

#### 方式2：直接使用 Python

```bash
python SPARQ/evaluate_test.py \
    --test_graph_query_path /root/autodl-tmp/dataset/CWQ/graph_query/graph_query_test.json \
    --test_original_data_path /root/autodl-tmp/dataset/CWQ/CWQ/test_simple.json \
    --entities_path /root/autodl-tmp/dataset/CWQ/CWQ/entities.txt \
    --relations_path /root/autodl-tmp/dataset/CWQ/CWQ/relations.txt \
    --checkpoint_path checkpoints/best.pt \
    --bert_model_path /root/autodl-tmp/bert-base-uncased \
    --beam_width 3 \
    --max_path_length 3 \
    --top_k 10 \
    --output_path output/test_results.json
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test_graph_query_path` | str | 必填 | 测试集图查询数据路径 |
| `--test_original_data_path` | str | 必填 | 测试集原始数据路径 |
| `--checkpoint_path` | str | 必填 | 训练好的模型检查点路径 |
| `--beam_width` | int | 3 | 束搜索宽度 |
| `--max_path_length` | int | 3 | 最大路径长度 |
| `--top_k` | int | 10 | 计算Hit@K的最大K值 |
| `--output_path` | str | output/test_results.json | 结果保存路径 |

### 输出结果示例

```
================================================================================
Evaluation Results
================================================================================
Total samples: 3537
Valid samples: 3500
Error samples: 37
--------------------------------------------------------------------------------
Hit@ 1:  15.23%
Hit@ 2:  22.45%
Hit@ 3:  28.67%
Hit@ 4:  33.12%
Hit@ 5:  37.89%
Hit@ 6:  41.23%
Hit@ 7:  44.56%
Hit@ 8:  47.34%
Hit@ 9:  49.87%
Hit@10:  52.15%
================================================================================
```

结果会保存到 JSON 文件中，包含：
- 评估配置参数
- Hit@1 ~ Hit@10 的精确数值
- 错误样本列表（用于调试）
