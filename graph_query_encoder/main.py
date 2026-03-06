"""
图结构查询编码器 - 主入口文件

仅支持推理/编码模式（不包含训练功能）
"""

import os
import argparse
import torch
import random
import numpy as np

from models import GraphQueryEncoder


def set_seed(seed: int):
    """设置随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="图结构查询编码器 - 推理/编码",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==================== 模型参数 ====================
    model_group = parser.add_argument_group("模型参数")
    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=768,
        help="隐藏层维度"
    )
    model_group.add_argument(
        "--num_gnn_layers",
        type=int,
        default=3,
        help="GNN层数"
    )
    model_group.add_argument(
        "--num_relations",
        type=int,
        default=100,
        help="关系类型数量"
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout概率"
    )
    model_group.add_argument(
        "--bert_model_name",
        type=str,
        default="bert-base-uncased",
        help="BERT模型名称或路径"
    )
    model_group.add_argument(
        "--use_bfs_gnn",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="是否使用BFS定向GNN (true/false)"
    )
    model_group.add_argument(
        "--use_query_centered_pooling",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="是否使用以'?'为中心的池化 (true/false)"
    )

    # ==================== 其他参数 ====================
    other_group = parser.add_argument_group("其他参数")
    other_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="运行设备"
    )
    other_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    other_group.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="模型检查点路径（可选）"
    )

    return parser.parse_args()


def encode(args):
    """编码模式 - 对示例查询进行编码"""
    print("=" * 60)
    print("图结构查询编码器")
    print("=" * 60)

    # 创建模型
    print("\n初始化模型...")
    encoder = GraphQueryEncoder(
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_relations=args.num_relations,
        bert_model_name=args.bert_model_name,
        dropout=args.dropout,
        use_bfs_gnn=args.use_bfs_gnn,
        use_query_centered_pooling=args.use_query_centered_pooling
    )

    # 加载检查点（如果提供）
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"加载检查点: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        encoder.load_state_dict(checkpoint["model_state_dict"])

    encoder.to(args.device)
    encoder.eval()

    # 示例查询
    examples = [
        {
            "question": "Who was the president in 1980 of the country that has Azad Kashmir?",
            "triples": [
                ["?", "president", "country"],
                ["country", "has", "Azad Kashmir"]
            ]
        },
        {
            "question": "What is the mascot of the team that has Nicholas S. Zeppos as its leader?",
            "triples": [
                ["team", "leader", "Nicholas S. Zeppos"],
                ["team", "mascot", "?"]
            ]
        }
    ]

    print("\n运行示例查询编码...")
    with torch.no_grad():
        for i, example in enumerate(examples, 1):
            print(f"\n示例 {i}:")
            print(f"  问题: {example['question']}")
            print(f"  三元组: {example['triples']}")

            h_eq = encoder(example["triples"])
            print(f"  查询表征维度: {h_eq.shape}")
            print(f"  表征前5维: {h_eq[:5].cpu().numpy()}")

    # 批量编码示例
    print("\n" + "-" * 40)
    print("批量编码示例")
    print("-" * 40)

    batch_triples = [ex["triples"] for ex in examples]
    with torch.no_grad():
        h_eq_batch = encoder.encode_batch(batch_triples)
    print(f"批量查询表征维度: {h_eq_batch.shape}")  # (2, hidden_dim)

    print("\n" + "=" * 60)
    print("编码完成!")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 运行编码
    encode(args)


if __name__ == "__main__":
    main()
