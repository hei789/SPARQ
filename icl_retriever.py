#!/usr/bin/env python3
"""
ICL检索模块：基于语义+实体+结构三维度匹配的相似样例检索
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ICLRetriever:
    """上下文学习样例检索器"""

    def __init__(self, examples_file: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化检索器

        Args:
            examples_file: 样例数据文件路径（由example_builder生成）
            embedding_model_name: 句向量模型名称
        """
        self.examples_file = examples_file
        self.embedding_model_name = embedding_model_name
        self.examples: List[Dict] = []
        self.question_embeddings: np.ndarray = None
        self.embedding_model = None

        self._load_examples()
        self._load_embedding_model()
        self._precompute_embeddings()

    def _load_examples(self):
        """加载样例数据"""
        with open(self.examples_file, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)
        print(f"加载了 {len(self.examples)} 个样例")

    def _load_embedding_model(self):
        """加载句向量编码模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"加载Embedding模型: {self.embedding_model_name}")
        except ImportError:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")

    def _precompute_embeddings(self):
        """预计算所有样例问题的句向量"""
        questions = [ex['question'] for ex in self.examples]
        self.question_embeddings = self.embedding_model.encode(questions, show_progress_bar=True)
        print(f"预计算完成，embedding形状: {self.question_embeddings.shape}")

    def encode_question(self, question: str) -> np.ndarray:
        """编码单个问题"""
        return self.embedding_model.encode([question])[0]

    def compute_semantic_similarity(self, query_emb: np.ndarray, example_idx: int) -> float:
        """
        计算语义相似度（余弦相似度）

        Weight: 0.5
        """
        example_emb = self.question_embeddings[example_idx].reshape(1, -1)
        query_emb = query_emb.reshape(1, -1)
        sim = cosine_similarity(query_emb, example_emb)[0][0]
        return float(sim)

    def compute_entity_overlap(self, query_entities: set, example_entities: list) -> float:
        """
        计算实体重叠度（Jaccard系数）

        Weight: 0.3
        """
        if not query_entities or not example_entities:
            return 0.0

        example_entities = set(example_entities)

        intersection = len(query_entities & example_entities)
        union = len(query_entities | example_entities)

        if union == 0:
            return 0.0

        return intersection / union

    def extract_question_structure(self, question: str) -> Dict:
        """
        提取问题结构特征

        Returns:
            {
                'wh_word': Wh词 (what/where/who/when/which/how...)
                'main_verb': 主要动词
                'entity_count': 实体数量估计
            }
        """
        question_lower = question.lower()

        # 提取Wh词
        wh_words = ['what', 'where', 'who', 'when', 'which', 'how', 'whom', 'whose']
        wh_word = None
        for w in wh_words:
            if w in question_lower:
                wh_word = w
                break

        # 提取主要动词（简单规则：第一个非be动词）
        be_verbs = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'did', 'do', 'does', 'have', 'has', 'had'}
        words = re.findall(r'\b[a-zA-Z]+\b', question_lower)
        main_verb = None
        for word in words[1:]:  # 跳过第一个词（通常是Wh词）
            if word not in be_verbs and len(word) > 2:
                main_verb = word
                break

        # 估计实体数量（大写单词数量）
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]*\b', question)
        entity_count = len(capitalized)

        return {
            'wh_word': wh_word,
            'main_verb': main_verb,
            'entity_count': entity_count
        }

    def compute_structure_similarity(self, query_struct: Dict, example_question: str) -> float:
        """
        计算结构相似度

        Weight: 0.2
        """
        example_struct = self.extract_question_structure(example_question)

        score = 0.0

        # Wh词匹配 (0.5权重)
        if query_struct['wh_word'] and example_struct['wh_word']:
            if query_struct['wh_word'] == example_struct['wh_word']:
                score += 0.5

        # 主要动词匹配 (0.3权重)
        if query_struct['main_verb'] and example_struct['main_verb']:
            if query_struct['main_verb'] == example_struct['main_verb']:
                score += 0.3
            # 可以考虑使用词向量计算动词相似度

        # 实体数量相似度 (0.2权重)
        entity_diff = abs(query_struct['entity_count'] - example_struct['entity_count'])
        entity_sim = max(0, 1 - entity_diff / max(query_struct['entity_count'], 1))
        score += 0.2 * entity_sim

        return score

    def retrieve(self, question: str, entities: Optional[List] = None,
                 top_k: int = 3, weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        检索最相似的样例

        Args:
            question: 查询问题
            entities: 查询问题中的实体列表
            top_k: 返回样例数量
            weights: 各维度权重，默认{'semantic': 0.5, 'entity': 0.3, 'structure': 0.2}

        Returns:
            相似样例列表（包含相似度分数）
        """
        if weights is None:
            weights = {'semantic': 0.5, 'entity': 0.3, 'structure': 0.2}

        # 编码查询问题
        query_emb = self.encode_question(question)
        query_entities = set(entities) if entities else set()
        query_struct = self.extract_question_structure(question)

        # 计算每个样例的综合相似度
        scores = []

        for idx, example in enumerate(self.examples):
            # 语义相似度
            sem_sim = self.compute_semantic_similarity(query_emb, idx)

            # 实体重叠度
            entity_sim = self.compute_entity_overlap(query_entities, example.get('entities', []))

            # 结构相似度
            struct_sim = self.compute_structure_similarity(query_struct, example['question'])

            # 综合得分
            total_score = (
                weights['semantic'] * sem_sim +
                weights['entity'] * entity_sim +
                weights['structure'] * struct_sim
            )

            scores.append({
                'idx': idx,
                'example': example,
                'semantic_sim': sem_sim,
                'entity_sim': entity_sim,
                'structure_sim': struct_sim,
                'total_score': total_score
            })

        # 按总分排序
        scores.sort(key=lambda x: x['total_score'], reverse=True)

        # 返回top_k
        return scores[:top_k]

    def format_icl_prompt(self, question: str, entities: Optional[List] = None,
                          top_k: int = 3, include_analysis: bool = False) -> str:
        """
        格式化ICL提示（包含检索到的相似样例）

        Args:
            question: 目标问题
            entities: 实体列表
            top_k: 样例数量
            include_analysis: 是否包含详细分析信息

        Returns:
            格式化的提示文本
        """
        similar_examples = self.retrieve(question, entities, top_k)

        prompt_parts = [
            "# 你是一个知识图谱问答专家，擅长将问题分析为结构化的三元组。",
            "",
            "# 请参考以下相似的问题分析样例：",
            ""
        ]

        for i, item in enumerate(similar_examples, 1):
            ex = item['example']
            prompt_parts.append(f"## 样例 {i}")
            prompt_parts.append(f"Question: {ex['question']}")

            if 'triples' in ex and ex['triples']:
                prompt_parts.append("Triples:")
                for triple in ex['triples']:
                    prompt_parts.append(f"({triple[0]}, {triple[1]}, {triple[2]})")

            if include_analysis:
                prompt_parts.append(f"[相似度分析: 语义={item['semantic_sim']:.3f}, "
                                  f"实体={item['entity_sim']:.3f}, "
                                  f"结构={item['structure_sim']:.3f}, "
                                  f"总分={item['total_score']:.3f}]")

            prompt_parts.append("")

        prompt_parts.extend([
            "# 现在请分析以下问题：",
            f"Question: {question}",
            "Triples:"
        ])

        return "\n".join(prompt_parts)

    def evaluate_retrieval(self, test_questions: List[str], test_entities: List[List],
                          ground_truth_indices: List[int]) -> Dict:
        """
        评估检索效果

        Args:
            test_questions: 测试问题列表
            test_entities: 测试问题实体列表
            ground_truth_indices: 每个测试问题的正确答案在examples中的索引

        Returns:
            评估指标
        """
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        mrr_sum = 0.0

        for question, entities, gt_idx in zip(test_questions, test_entities, ground_truth_indices):
            results = self.retrieve(question, entities, top_k=10)
            retrieved_indices = [r['idx'] for r in results]

            if gt_idx in retrieved_indices[:1]:
                hits_at_1 += 1
            if gt_idx in retrieved_indices[:3]:
                hits_at_3 += 1
            if gt_idx in retrieved_indices[:5]:
                hits_at_5 += 1

            if gt_idx in retrieved_indices:
                rank = retrieved_indices.index(gt_idx) + 1
                mrr_sum += 1.0 / rank

        n = len(test_questions)
        return {
            'hits@1': hits_at_1 / n,
            'hits@3': hits_at_3 / n,
            'hits@5': hits_at_5 / n,
            'mrr': mrr_sum / n
        }


if __name__ == "__main__":
    """示例用法"""
    EXAMPLES_FILE = "dataset/processed/examples_with_triples.json"
    retriever = ICLRetriever(EXAMPLES_FILE)

    test_question = "Where did the artist that recorded Wedding Day go to college?"
    test_entities = [12345, 67890]

    results = retriever.retrieve(test_question, test_entities, top_k=3)

    for i, item in enumerate(results, 1):
        ex = item['example']
        print(f"[Top {i}] 相似度: {item['total_score']:.4f}")
        print(f"  问题: {ex['question']}")
        print(f"  三元组: {ex.get('triples', [])}")
