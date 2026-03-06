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

## 删除的非必要文件

- `run_encoder.py` - 可用 `graph_query_encoder/main.py` 替代
- `run_retriever.py` - 可用 `graph_retriever/main.py` 或根目录 `main.py` 替代
- `CHANGES.md` - 变更日志
- `USAGE.md` - 使用说明（内容合并到README.md）
- `graph_query_encoder/example_usage.py` - 示例代码
- `graph_query_encoder/retriever_example.py` - 示例代码
- `graph_query_encoder/run_retriever.py` - 重复入口
- `graph_query_encoder/config.py` - 配置文件（配置已移至main.py参数）
- `graph_query_encoder/README.md` - 模块文档
- `graph_retriever/example_usage.py` - 示例代码
- `graph_retriever/README.md` - 模块文档

## 使用方式

### 1. 完整流程（推荐）
```bash
python main.py \
    --graph_query_path dataset/CWQ/CWQ/graph_query_dev.json \
    --original_data_path dataset/CWQ/CWQ/dev_simple.json \
    --entities_path dataset/CWQ/CWQ/entities.txt \
    --relations_path dataset/CWQ/CWQ/relations.txt
```

### 2. ICL检索
```python
from icl_retriever import ICLRetriever
retriever = ICLRetriever("examples.json")
results = retriever.retrieve("问题文本", entities=[...], top_k=3)
```

## 代码简化说明

- 删除了详细的docstring和注释
- 删除了示例代码和测试代码
- 删除了多余的命令行入口
- 合并了重复的配置定义
- 保留了核心算法实现
