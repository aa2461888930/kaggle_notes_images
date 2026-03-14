from ranx import Qrels, Run, evaluate




# ==========================================
# 1. 准备标准答案 (Qrels - Ground Truth)
# ==========================================
# 格式要求：{ query_id: { doc_id: 相关度分数 } }
# 对于二元相关性（相关为1，不相关为0或不写），通常分数设为 1
qrels_dict = {
    "query_01": {"Pharmaceutical_Polymeric_Materials,chunk_10": 1, "Pharmaceutical_Polymeric_Materials,chunk_12": 1},
    "query_02": {"Pharmacology_Liu_Xiaodong_5th_Edition_2019,chunk_55": 1},
    "query_03": {"Medicinal_Chemistry_4th_Edition_You_Qidong,chunk_08": 1}
}

# 将字典转换为 ranx 的 Qrels 对象
qrels = Qrels(qrels_dict)


# ==========================================
# 2. 获取检索结果 (Run - Retrieval Output)
# ==========================================
# 格式要求：{ query_id: { doc_id: 检索相似度得分 } }
# 分数越高，代表检索系统认为它越相关，排在越前面。

run_dict = {}

# 💡 假设 queries_list 是你的测试问题列表: ["高分子材料的特点是什么？", ...]
# 下面是你需要接入 ChromaDB 的伪代码逻辑：

"""
for q_id, query_text in queries_list:
    # 使用你之前建好的 Chroma client 进行查询
    results = collection.query(
        query_texts=[query_text],
        n_results=5  # 比如检索前 Top-5
    )
    
    # 构建当前 query 的结果字典
    current_query_results = {}
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        # 注意：Chroma 默认返回的是距离 (distance)，越小越相似。
        # ranx 期望的是分数 (score)，越大越好。所以如果是 L2 距离，可以用 1 / (1 + distance) 转换。
        # 如果你 Chroma 用的是 cosine 相似度，直接用即可。
        score = 1.0 / (1.0 + results['distances'][0][i]) 
        current_query_results[doc_id] = score
        
    run_dict[q_id] = current_query_results
"""

# 为了演示，我们手动模拟 Chroma 返回的 Run 字典
run_dict = {
    "query_01": {
        "Pharmaceutical_Polymeric_Materials,chunk_10": 0.95, # 完美命中第一个
        "Pharmaceutical_Polymeric_Materials,chunk_99": 0.88, # 误报的干扰项
        "Pharmaceutical_Polymeric_Materials,chunk_12": 0.70  # 命中第二个，但排名靠后
    },
    "query_02": {
        "Pharmacology_Liu_Xiaodong_5th_Edition_2019,chunk_99": 0.90, # 排第一的是误报
        "Pharmacology_Liu_Xiaodong_5th_Edition_2019,chunk_55": 0.85  # 真实答案排第二
    },
    "query_03": {
        "Medicinal_Chemistry_4th_Edition_You_Qidong,chunk_08": 0.99  # 完美命中
    }
}

# 将字典转换为 ranx 的 Run 对象
run = Run(run_dict)


# ==========================================
# 3. 执行评估并输出结果
# ==========================================
# 选择信息检索常用的几个核心指标：
# MRR (Mean Reciprocal Rank): 评估第一个正确答案出现的位置（越靠前越好）
# NDCG@5: 评估 Top-5 的综合排序质量
# Recall@5: 评估 Top-5 到底找回了多少个标准答案

metrics = ["mrr", "ndcg@5", "recall@5"]
results = evaluate(qrels, run, metrics=metrics)

print("🏆 检索系统评估结果:")
for metric, score in results.items():
    print(f"{metric.upper()}: {score:.4f}")

# 如果你需要导出详细报告
# print(results)