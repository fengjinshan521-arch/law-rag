import csv
import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_common import build_llm

print("Step 5：加载评分结果（无需重建向量库）...")

JUDGE_PROMPT = """你是法律问答系统的自动评测裁判。请基于给定的材料，对【回答】打分。

【用户问题】
{question}

【检索到的法条片段】
{context}

【标准答案】
{ground_truth}

【待评测的回答】
{answer}

请严格按以下标准打 0~1 的小数分（可保留一位小数）：
1. faithfulness（忠实度）：回答中的每一句是否都能在【检索到的法条片段】中找到依据。完全有依据=1.0，出现检索片段之外的编造内容则扣分，全部编造=0.0。
2. answer_relevancy（答案相关性）：回答是否正面、完整地回应了用户问题。答非所问或只回"未找到"而片段中其实有答案=0.0。
3. context_recall（上下文召回）：【检索到的法条片段】是否覆盖了【标准答案】的关键要点。全部覆盖=1.0，部分覆盖=0.5，完全没覆盖=0.0。

只输出一个 JSON 对象，不要有任何多余文字、不要代码块：
{{"faithfulness": 分数, "answer_relevancy": 分数, "context_recall": 分数, "reason": "一句话说明理由"}}
"""


def parse_score(text: str) -> dict:
    """从裁判输出里抠出 JSON，容错代码块、前后废话。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {"faithfulness": None, "answer_relevancy": None, "context_recall": None,
                "reason": f"裁判输出无法解析：{text[:100]}"}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"faithfulness": None, "answer_relevancy": None, "context_recall": None,
                "reason": f"JSON 解析失败：{m.group(0)[:100]}"}
    return {
        "faithfulness": data.get("faithfulness"),
        "answer_relevancy": data.get("answer_relevancy"),
        "context_recall": data.get("context_recall"),
        "reason": data.get("reason", ""),
    }


def to_float(v):
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


def main():
    print("=" * 60)
    print("Step 5：LLM 裁判自动评分")
    print("=" * 60)

    try:
        with open("eval_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("找不到 eval_results.json，请先运行 04_evaluate.py")
        return

    llm = build_llm()
    judge = ChatPromptTemplate.from_template(JUDGE_PROMPT) | llm | StrOutputParser()

    scored = []
    for i, r in enumerate(results, 1):
        ctx = "\n\n".join(d["text"] for d in r.get("retrieved", [])) or "（无检索结果）"
        print(f"  [{i}/{len(results)}] 评分中：{r['query'][:20]}...")
        raw = judge.invoke({
            "question": r["query"],
            "context": ctx,
            "ground_truth": r["ground_truth"],
            "answer": r["rag_answer"],
        })
        s = parse_score(raw)
        scored.append({
            "query": r["query"],
            "faithfulness": to_float(s["faithfulness"]),
            "answer_relevancy": to_float(s["answer_relevancy"]),
            "context_recall": to_float(s["context_recall"]),
            "reason": s["reason"],
        })

    with open("eval_scores.json", "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)

    with open("eval_scores.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "faithfulness", "answer_relevancy", "context_recall", "reason"])
        w.writeheader()
        w.writerows(scored)

    def avg(key):
        vals = [s[key] for s in scored if s[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"  忠实度 faithfulness     : {avg('faithfulness')}")
    print(f"  答案相关性 relevancy    : {avg('answer_relevancy')}")
    print(f"  上下文召回 recall       : {avg('context_recall')}")
    print("\n明细已保存：eval_scores.json / eval_scores.csv")


if __name__ == "__main__":
    main()