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
├── TRAINING.md           # 本文件
└── checkpoints/          # 检查点保存目录
    ├── best.pt           # 最佳模型
    ├── latest.pt         # 最新模型
    └── epoch_*.pt        # 每轮模型
```
