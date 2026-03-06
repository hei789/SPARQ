"""
图检索器主入口
支持从CWQ数据集加载数据并执行检索

所有路径通过命令行参数传入，便于配置
"""

import sys
import os

# 添加父目录到路径以导入 graph_query_encoder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import json
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

from graph_retriever import (
    GraphRetriever,
    IntegratedRetriever,
    CWQDataLoader,
    SubgraphBuilder
)
from graph_query_encoder.models import GraphQueryEncoder


class RetrieverRunner:
    """检索器运行器"""

    def __init__(
        self,
        query_encoder: GraphQueryEncoder,
        retriever: GraphRetriever,
        device: torch.device = None
    ):
        self.query_encoder = query_encoder
        self.retriever = retriever
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.integrated = IntegratedRetriever(
            query_encoder=query_encoder,
            retriever=retriever,
            device=self.device
        ).to(self.device)

    def retrieve_for_sample(
        self,
        sample: Dict[str, Any],
        subgraph_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        为单个样本执行检索

        Args:
            sample: 图查询样本
            subgraph_data: 子图数据（可选）

        Returns:
            检索结果
        """
        triples = sample.get("triples", [])

        if not triples:
            return {"error": "No triples in sample"}

        # 如果有子图数据，使用它
        if subgraph_data and "tuples" in subgraph_data:
            # 从原始数据构建子图
            node_features, edge_index, edge_types, entity2idx = \
                SubgraphBuilder.build_from_tuples(
                    subgraph_data["tuples"],
                    subgraph_data.get("entities", []),
                    self.retriever.hidden_dim,
                    self.device
                )
        else:
            # 构建模拟子图
            num_nodes = 20
            node_features = torch.randn(
                num_nodes, self.retriever.hidden_dim, device=self.device
            )
            edge_index = torch.randint(
                0, num_nodes, (2, 50), device=self.device
            )
            edge_types = torch.randint(
                0, 10, (50,), device=self.device
            )
            entity2idx = {f"entity_{i}": i for i in range(num_nodes)}

        # 执行检索
        self.integrated.eval()
        with torch.no_grad():
            result = self.integrated(
                triples=triples,
                subgraph_node_features=node_features,
                subgraph_edge_index=edge_index,
                subgraph_edge_types=edge_types,
                entity2idx=entity2idx,
                top_k=10
            )

        return result

    def retrieve_batch(
        self,
        samples: List[Dict[str, Any]],
        subgraphs: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量检索

        Args:
            samples: 样本列表
            subgraphs: 子图数据列表（可选）

        Returns:
            检索结果列表
        """
        results = []
        for i, sample in enumerate(samples):
            subgraph = subgraphs[i] if subgraphs and i < len(subgraphs) else None
            result = self.retrieve_for_sample(sample, subgraph)
            results.append(result)
        return results


def parse_args():
    """解析命令行参数 - 所有路径都在此定义"""
    parser = argparse.ArgumentParser(
        description="图神经网络检索器 - 所有路径通过参数传入",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==================== 数据路径参数 (在入口处定义) ====================
    path_group = parser.add_argument_group("数据路径参数 (必需)")
    path_group.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="图查询数据文件路径 (如: dataset/CWQ/CWQ/graph_query_dev.json)"
    )
    path_group.add_argument(
        "--entities_path",
        type=str,
        required=True,
        help="实体列表文件路径 (如: dataset/CWQ/CWQ/entities.txt)"
    )
    path_group.add_argument(
        "--relations_path",
        type=str,
        required=True,
        help="关系列表文件路径 (如: dataset/CWQ/CWQ/relations.txt)"
    )
    path_group.add_argument(
        "--original_data_path",
        type=str,
        default=None,
        help="原始数据路径 (可选，用于加载子图，如: dataset/CWQ/CWQ/dev_simple.json)"
    )

    # ==================== 模型配置 ====================
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=768,
        help="隐藏层维度"
    )
    model_group.add_argument(
        "--num_query_layers",
        type=int,
        default=3,
        help="查询编码器GNN层数"
    )
    model_group.add_argument(
        "--num_retriever_layers",
        type=int,
        default=2,
        help="检索器GNN层数"
    )
    model_group.add_argument(
        "--num_relations",
        type=int,
        default=100,
        help="关系类型数量"
    )

    # ==================== 检索配置 ====================
    retrieval_group = parser.add_argument_group("检索配置")
    retrieval_group.add_argument(
        "--beam_width",
        type=int,
        default=3,
        help="束搜索宽度"
    )
    retrieval_group.add_argument(
        "--max_path_length",
        type=int,
        default=3,
        help="最大路径长度"
    )
    retrieval_group.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.0,
        help="相似度阈值"
    )
    retrieval_group.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="返回的路径数量"
    )

    # ==================== 运行配置 ====================
    run_group = parser.add_argument_group("运行配置")
    run_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数"
    )
    run_group.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出结果路径 (如: output/retrieval_results.json)"
    )
    run_group.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="模型检查点路径 (可选)"
    )
    run_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="运行设备"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 70)
    print("图神经网络检索器")
    print("=" * 70)

    # 打印数据路径
    print("\n【数据路径配置】")
    print(f"  图查询数据: {args.data_path}")
    print(f"  实体列表:   {args.entities_path}")
    print(f"  关系列表:   {args.relations_path}")
    if args.original_data_path:
        print(f"  原始数据:   {args.original_data_path}")

    print("\n【模型配置】")
    print(f"  隐藏层维度: {args.hidden_dim}")
    print(f"  查询编码器层数: {args.num_query_layers}")
    print(f"  检索器层数: {args.num_retriever_layers}")

    print("\n【检索配置】")
    print(f"  束宽度: {args.beam_width}")
    print(f"  最大路径长度: {args.max_path_length}")
    print(f"  Top-K: {args.top_k}")
    print("=" * 70)

    # 设备
    device = torch.device(args.device)
    print(f"\n使用设备: {device}")

    # 加载数据
    try:
        data_loader = CWQDataLoader(
            data_path=args.data_path,
            entities_path=args.entities_path,
            relations_path=args.relations_path
        )
        samples = data_loader.load_graph_queries()
        print(f"\n成功加载 {len(samples)} 个样本")
    except FileNotFoundError as e:
        print(f"\n数据加载失败: {e}")
        print("请检查数据路径是否正确。")
        return

    # 创建模型
    query_encoder = GraphQueryEncoder(
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_query_layers,
        num_relations=args.num_relations,
        use_query_centered_pooling=True,
        use_bfs_gnn=True
    ).to(device)

    retriever = GraphRetriever(
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_retriever_layers,
        num_relations=args.num_relations,
        beam_width=args.beam_width,
        max_path_length=args.max_path_length,
        similarity_threshold=args.similarity_threshold
    ).to(device)

    # 加载检查点（如果提供）
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"\n加载检查点: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if "query_encoder_state" in checkpoint:
            query_encoder.load_state_dict(checkpoint["query_encoder_state"])
        if "retriever_state" in checkpoint:
            retriever.load_state_dict(checkpoint["retriever_state"])

    print("\n模型创建完成:")
    print(f"  查询编码器参数: {sum(p.numel() for p in query_encoder.parameters()):,}")
    print(f"  检索器参数: {sum(p.numel() for p in retriever.parameters()):,}")

    # 运行检索
    runner = RetrieverRunner(query_encoder, retriever, device)

    # 限制样本数
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"\n开始检索 {len(samples)} 个样本...")

    all_results = []
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"  处理 {i}/{len(samples)}...")

        result = runner.retrieve_for_sample(sample)
        all_results.append({
            "id": sample.get("id", ""),
            "question": sample.get("question", ""),
            "num_paths": len(result.get("paths", [])),
            "topic_entities": result.get("topic_entities", []),
            "top_similarities": result.get("similarities", []).tolist()
            if torch.is_tensor(result.get("similarities"))
            else []
        })

    print(f"\n检索完成!")

    # 统计结果
    total_paths = sum(r["num_paths"] for r in all_results)
    avg_paths = total_paths / len(all_results) if all_results else 0
    print(f"平均每个样本检索到 {avg_paths:.2f} 条路径")

    # 保存结果
    if args.output_path:
        # 确保输出目录存在
        output_dir = Path(args.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {args.output_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
