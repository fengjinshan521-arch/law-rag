# 04_evaluate.py —— 批量跑测试集，产出答案与检索证据
#
# 【原报错】base_url 写的是 https://api.siliconflow.cn/v1，
#           但本地只配置了 DASHSCOPE_API_KEY（没有 SILICONFLOW_API_KEY），
#           调用必然 401 / Connection error。
# 【修复】  统一走 rag_common 的 DashScope 端点。
#
# 运行：python 04_evaluate.py   -> 输出 eval_results.json
import json

from rag_common import EVAL_SET, build_rag_chain, load_vector_store

print("=" * 60)
print("Step 4：批量评测")
print("=" * 60)

store = load_vector_store()
chain, retriever, _ = build_rag_chain(k=3, store=store)

with open(EVAL_SET, "r", encoding="utf-8") as f:
    test_cases = [json.loads(line) for line in f if line.strip()]
print(f"加载 {len(test_cases)} 条测试用例，开始生成答案...\n")

results = []
for i, case in enumerate(test_cases, 1):
    q = case["query"]
    print(f"  [{i}/{len(test_cases)}] {q[:24]}...")

    docs = retriever.invoke(q)
    try:
        answer = chain.invoke(q)
    except Exception as e:
        answer = f"[调用失败] {type(e).__name__}: {e}"
        print(f"      ! {answer}")

    results.append({
        "query": q,
        "ground_truth": case["ground_truth"],
        "rag_answer": answer,
        "retrieved": [
            {"source": d.metadata.get("law_name", ""), "score": round(d.metadata.get("score", 0), 4),
             "text": d.page_content[:300]}
            for d in docs
        ],
    })

with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n完成！结果已保存至 eval_results.json（共 {len(results)} 条）")
print("下一步：运行 05_ragas_eval.py 用大模型自动打分。")
