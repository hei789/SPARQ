# SPARQ 代码结构说明

## 保留的核心文件

### 根目录
- `main.py` - GraphRAG整体流程入口（简化后约260行）
- `icl_retriever.py` - ICL检索模块（用于检索相似样例）
- `README.md` - 项目说明文档

### graph_query_encoder/ - 图查询编码器模块
核心文件保留：
- `main.py` - 编码器入口
- `models/__init__.py` - 模型导出
- `models/graph_query_encoder.py` - 图查询编码器核心
- `models/graph_retriever.py` - 图检索器模型
- `models/embedding_layer.py` - 嵌入层
- `models/gnn_layers.py` - GNN层实现
- `data/__init__.py` - 数据模块
- `data/dataset.py` - 数据集定义

### graph_retriever/ - 图检索器模块
核心文件保留：
- `__init__.py` - 模块导出
- `main.py` - 检索器入口
- `graph_retriever_core.py` - 检索器核心实现
- `data_loader.py` - CWQ数据加载器

## 使用方式

### 1. 完整流程（推荐）
```bash
python SPARQ/main.py \
    --graph_query_path /root/autodl-tmp/dataset/CWQ/graph_query/graph_query_dev.json \
    --original_data_path /root/autodl-tmp/dataset/CWQ/CWQ/dev_simple.json \
    --entities_path /root/autodl-tmp/dataset/CWQ/CWQ/entities.txt \
    --relations_path /root/autodl-tmp/dataset/CWQ/CWQ/relations.txt \
    --bert_model_path /root/autodl-tmp/bert-base-uncased \
    --output_path output/results.json
```

### 2. ICL检索
```python
from icl_retriever import ICLRetriever
retriever = ICLRetriever("examples.json")
results = retriever.retrieve("问题文本", entities=[...], top_k=3)
```
