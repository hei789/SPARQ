"""
数据集加载和处理

仅用于编码器的数据加载（不包含训练相关功能）
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import os


class GraphQueryDataset(Dataset):
    """
    图结构查询数据集
    用于批量编码图结构查询
    """

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_path: 数据文件路径（JSON格式）
            max_samples: 最大样本数（用于调试）
        """
        self.data_path = data_path
        self.samples = self._load_data()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本

        Returns:
            sample: 包含以下字段的字典
                - id: 样本ID
                - question: 问题文本
                - triples: 三元组列表
                - triples_text: 三元组的文本表示
        """
        sample = self.samples[idx]

        return {
            "id": sample.get("id", ""),
            "question": sample.get("question", ""),
            "triples": sample.get("triples", []),
            "triples_text": sample.get("triples_text", ""),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    批处理函数

    Args:
        batch: 样本列表

    Returns:
        batched_data: 批处理后的数据
    """
    return {
        "ids": [sample["id"] for sample in batch],
        "questions": [sample["question"] for sample in batch],
        "triples_list": [sample["triples"] for sample in batch],
        "triples_texts": [sample["triples_text"] for sample in batch],
    }
