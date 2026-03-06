"""
节点和关系的初始特征编码层
使用BERT_base进行编码
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Optional


class EmbeddingLayer(nn.Module):
    """
    使用BERT编码节点（实体）和关系的初始特征
    """

    def __init__(self, bert_model_name: str = "bert-base-uncased", hidden_dim: int = 768):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # 特殊标记的嵌入
        self.query_embedding = nn.Parameter(torch.randn(1, hidden_dim))

        # 如果BERT输出维度与目标维度不同，添加投影层
        if self.bert.config.hidden_size != hidden_dim:
            self.projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        else:
            self.projection = None

    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        使用BERT编码文本列表

        Args:
            texts: 文本列表
            device: 计算设备

        Returns:
            embeddings: (num_texts, hidden_dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # 移动到设备
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # BERT编码
        with torch.no_grad():  # 可选：是否微调BERT
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 使用[CLS]token的表示
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, bert_dim)

        # 投影到目标维度
        if self.projection is not None:
            cls_embeddings = self.projection(cls_embeddings)

        return cls_embeddings

    def encode_entities(self, entities: List[str], device: torch.device) -> torch.Tensor:
        """
        编码实体列表

        Args:
            entities: 实体名称列表
            device: 计算设备

        Returns:
            embeddings: (num_entities, hidden_dim)
        """
        # 处理"?"标记
        processed_entities = []
        query_indices = []

        for i, entity in enumerate(entities):
            if entity == "?":
                query_indices.append(i)
                processed_entities.append("[unused0]")  # 使用未使用的token占位
            else:
                processed_entities.append(entity)

        if len(processed_entities) == 0:
            return torch.zeros(0, self.hidden_dim, device=device)

        # 编码实体
        embeddings = self.encode_texts(processed_entities, device)

        # 替换"?"的嵌入
        if query_indices:
            for idx in query_indices:
                embeddings[idx] = self.query_embedding.squeeze(0)

        return embeddings

    def encode_relations(self, relations: List[str], device: torch.device) -> torch.Tensor:
        """
        编码关系列表

        Args:
            relations: 关系名称列表
            device: 计算设备

        Returns:
            embeddings: (num_relations, hidden_dim)
        """
        if len(relations) == 0:
            return torch.zeros(0, self.hidden_dim, device=device)

        # 将关系转换为可读文本（例如：将下划线替换为空格）
        processed_relations = [r.replace("_", " ") for r in relations]

        return self.encode_texts(processed_relations, device)

    def forward(
        self,
        entities: List[str],
        relations: List[str],
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            entities: 实体列表
            relations: 关系列表
            device: 计算设备

        Returns:
            entity_embeddings: (num_entities, hidden_dim)
            relation_embeddings: (num_relations, hidden_dim)
        """
        entity_embeddings = self.encode_entities(entities, device)
        relation_embeddings = self.encode_relations(relations, device)

        return entity_embeddings, relation_embeddings
