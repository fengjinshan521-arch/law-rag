# 07_auto_workflow.py —— 自动化评测工作流
#
# 【它做什么】一条命令完成"批量问答 -> 自动打分 -> 生成报告 -> 和上次对比"：
#   1. 读取 eval_set.jsonl 里的测试题
#   2. 每一题都跑一遍 06 的多 Agent 流水线
#   3. 自动算两个硬指标：
#        命中率  = 标准答案里提到的法条，有没有被检索到（客观事实，不需要模型判断）
#        依据分  = Agent 4 校验给出的 grounded 得分（查幻觉）
#   4. 生成 reports/YYYY-MM-DD.md 报告
#   5. 结果追加进 workflow_history.json，自动和上一次对比
#
# 【怎么跑】
#   python 07_auto_workflow.py           跑全部测试题
#   python 07_auto_workflow.py 3         只跑前 3 题（调试用，省时间）
#
# 【为什么要自动化】改了切分方式、换了模型、加了法条之后，
#   不用再凭感觉判断"好像变好了"，跑一次就知道命中率从 0.75 掉到 0.62 还是涨到 0.88。
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from rag_common import EVAL_SET
from multi_agent_runner import run_pipeline

REPORT_DIR = Path(__file__).resolve().parent / "reports"
HISTORY_FILE = Path(__file__).resolve().parent / "workflow_history.json"

# 匹配中文条号，如「第一千零七十七条」「第一百八十八条」「第十九条」
ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千零〇\d]+条")


def extract_articles(text: str) -> list:
    """从标准答案里抽出提到的法条号。"""
    return list(dict.fromkeys(ARTICLE_RE.findall(text or "")))


def calc_hit(articles: list, sources: list) -> bool:
    """标准答案里的关键法条，是否被检索到了。"""
    if not articles:
        return False
    all_text = "".join(s["snippet"] for s in sources)
    return any(a in all_text for a in articles)


def calc_rank(articles: list, sources: list):
    """
    期望法条第一次出现的位置（从 1 开始），没出现返回 None。
    MRR = 1/rank 的平均值——命中率只回答"找没找到"，
    MRR 进一步回答"排得够不够靠前"。排第 1 位得 1.0，第 3 位只有 0.33。
    """
    for i, s in enumerate(sources, 1):
        if any(a in s["snippet"] for a in articles):
            return i
    return None


def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def build_report(results: dict, prev: dict | None) -> str:
    lines = [
        f"# law-rag 自动化评测报告",
        "",
        f"- 时间：{results['timestamp']}",
        f"- 测试题数：{results['total']}",
        f"- 知识库切片：{results['kb_size']} 条",
        "",
        "## 总体指标",
        "",
        "| 指标 | 本次 | 上次 | 变化 |",
        "|---|---|---|---|",
    ]

    def row(name, key):
        cur = results[key]
        old = prev.get(key) if prev else None
        if old is None:
            delta = "—"
        else:
            diff = cur - old
            delta = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
        old_str = f"{old:.3f}" if old is not None else "—"
        return f"| {name} | {cur:.3f} | {old_str} | {delta} |"

    lines.append(row("法条命中率", "hit_rate"))
    lines.append(row("MRR 排序质量", "mrr"))
    lines.append(row("平均依据分", "avg_score"))
    lines.append(row("有依据回答占比", "grounded_rate"))
    lines.append("")

    if results.get("by_law"):
        lines.append("## 按法律分组")
        lines.append("")
        lines.append("| 法律 | 题数 | 命中率 | MRR |")
        lines.append("|---|---|---|---|")
        for law, st in results["by_law"].items():
            lines.append(f"| {law} | {st['n']} | {st['hit']:.3f} | {st['mrr']:.3f} |")
        lines.append("")

    lines.append("## 逐题明细")
    lines.append("")
    lines.append("| # | 法律 | 问题 | 命中 | 排名 | 依据分 | 重写 |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(results["details"], 1):
        hit = "是" if r["hit"] else "否"
        rank = str(r["rank"]) if r["rank"] else "未命中"
        lines.append(
            f"| {i} | {r['law']} | {r['query'][:24]} | {hit} | {rank} | {r['score']:.2f} | {r['retry']} |"
        )
    lines.append("")

    if prev:
        lines.append(f"> 对比基准：{prev['timestamp']} 的那次运行。")
    else:
        lines.append("> 首次运行，暂无历史对比基准。")
    lines.append("")
    return "\n".join(lines)


def main():
    # 参数：python 07_auto_workflow.py [题数] [--set=eval_set_v2.jsonl]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(
        (a[2:].split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--")
    )
    limit = int(args[0]) if args and args[0].isdigit() else None
    eval_file = Path(opts.get("set") or "eval_set.jsonl")
    if not eval_file.is_absolute():
        eval_file = Path(__file__).resolve().parent / eval_file

    with open(eval_file, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if limit:
        cases = cases[:limit]

    print("=" * 60)
    print(f"自动化评测工作流启动，共 {len(cases)} 题（{eval_file.name}）")
    print("=" * 60)

    from rag_common import load_vector_store

    kb_size = len(load_vector_store())
    details = []

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['query'][:30]}...")
        result = run_pipeline(case["query"], verbose=False)
        # 优先用测试集里显式标注的期望条号，没有再从标准答案里正则提取
        articles = case.get("articles") or extract_articles(case["ground_truth"])
        hit = calc_hit(articles, result["sources"])
        rank = calc_rank(articles, result["sources"])
        details.append({
            "query": case["query"],
            "law": case.get("law", "未标注"),
            "hit": hit,
            "rank": rank,
            "articles": articles,
            "score": result["score"],
            "grounded": result["grounded"],
            "retry": result["retry"],
            "answer": result["answer"],
        })
        rank_str = f"第{rank}位" if rank else "未命中"
        print(f"   命中：{'是' if hit else '否'}（{rank_str}）  依据分：{result['score']:.2f}  重写：{result['retry']} 次")

    total = len(details)
    rr = [1.0 / d["rank"] if d["rank"] else 0.0 for d in details]

    # 按法律分组统计，能看出哪部法律的检索效果拖后腿
    by_law = {}
    for d in details:
        st = by_law.setdefault(d["law"], {"n": 0, "hit": 0.0, "rr": 0.0})
        st["n"] += 1
        st["hit"] += 1 if d["hit"] else 0
        st["rr"] += 1.0 / d["rank"] if d["rank"] else 0.0
    by_law = {
        k: {"n": v["n"], "hit": v["hit"] / v["n"], "mrr": v["rr"] / v["n"]}
        for k, v in by_law.items()
    }

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "kb_size": kb_size,
        "eval_set": eval_file.name,
        "hit_rate": sum(1 for d in details if d["hit"]) / total,
        "mrr": sum(rr) / total,
        "avg_score": sum(d["score"] for d in details) / total,
        "grounded_rate": sum(1 for d in details if d["grounded"]) / total,
        "by_law": by_law,
        "details": details,
    }

    history = load_history()
    prev = history[-1] if history else None
    history.append(summary)
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.md"
    report_path.write_text(build_report(summary, prev), encoding="utf-8")

    print("\n" + "=" * 60)
    print("本次结果")
    print("=" * 60)
    print(f"  法条命中率   : {summary['hit_rate']:.3f}")
    print(f"  MRR 排序质量 : {summary['mrr']:.3f}")
    print(f"  平均依据分   : {summary['avg_score']:.3f}")
    print(f"  有依据回答   : {summary['grounded_rate']:.3f}")
    print("\n  按法律分组：")
    for law, st in summary["by_law"].items():
        print(f"    {law:<8} 题数{st['n']:>3}  命中率 {st['hit']:.3f}  MRR {st['mrr']:.3f}")
    if prev:
        print(f"\n  对比上次（{prev['timestamp']}）：")
        print(f"  命中率 {prev['hit_rate']:.3f} -> {summary['hit_rate']:.3f}")
        print(f"  依据分 {prev['avg_score']:.3f} -> {summary['avg_score']:.3f}")
    print(f"\n报告已生成：{report_path}")
    print(f"历史记录：{HISTORY_FILE}（共 {len(history)} 次）")


if __name__ == "__main__":
    main()
