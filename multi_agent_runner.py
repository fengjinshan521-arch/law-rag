# multi_agent_runner.py —— 多 Agent 流水线核心实现（可被其他脚本 import）
#
# 【为什么单独放一个文件】06_multi_agent.py 以数字开头，不是合法的 Python 模块名，
#   没法被 `import`。所以要复用这套 Agent，必须放在这个正常命名的文件里。
#
# 【这是什么】把原来"一个大模型干所有事"拆成 4 个专职 Agent，顺序协作：
#   Agent 1 意图分析：把口语问题改写成适合检索的法律术语，并判断属于哪个法律领域
#   Agent 2 法条检索：向量检索 + 关键词加分，按相关度挑出法条片段
#   Agent 3 法律分析：只依据检索到的法条作答，并必须标注条号
#   Agent 4 合规校验：逐句检查答案是否都有法条依据（查幻觉），不通过就让 Agent 3 重写
#
# 【为什么这么拆】
#   单 Agent 时，"挑法条"和"写答案"混在一次调用里，模型容易自己脑补。
#   拆开后每一步的中间结果都能单独看见、单独调试、单独换模型。
#
# 【依赖】先跑过 02_vectorstore.py 建好向量库
from __future__ import annotations

import json
import re
import sys

from rag_common import build_llm, load_vector_store

# 相似度低于这个值就认为是噪声，直接丢弃（避免无关法条污染答案）
SCORE_THRESHOLD = 0.35
# 校验不通过时最多重写几轮
MAX_RETRY = 1

_llm = None
_store = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = build_llm()
    return _llm


def get_store():
    global _store
    if _store is None:
        _store = load_vector_store()
    return _store


# ============================================================
# 工具函数：从大模型输出里抠 JSON（模型经常夹带废话，必须容错）
# ============================================================
def extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def call_llm(prompt: str) -> str:
    return get_llm().invoke(prompt).content


# ============================================================
# Agent 1：意图分析
# ============================================================
def agent_intent(question: str) -> dict:
    """把口语化问题改写成法律检索词，并识别法律领域。"""
    prompt = f"""你是法律检索助手。请分析下面的用户问题，只输出一个 JSON 对象，不要任何多余文字。

用户问题：{question}

输出格式：
{{"domain": "所属法律领域，从 婚姻家庭/合同/刑事/劳动/侵权/继承/其他 中选一个",
  "keywords": ["3到5个用于检索的法律关键词"],
  "rewritten_query": "改写后的检索语句，用法律术语表达，便于在法条库中检索"}}
"""
    data = extract_json(call_llm(prompt))
    # 解析失败时降级：直接用原问题检索，保证流水线不中断
    return {
        "domain": data.get("domain", "其他"),
        "keywords": data.get("keywords", []) if isinstance(data.get("keywords"), list) else [],
        "rewritten_query": data.get("rewritten_query") or question,
    }


# ============================================================
# Agent 2：法条检索（工具型 Agent，不调用大模型）
# ============================================================
def agent_retrieve(query: str, keywords: list, k: int = 5) -> list:
    """
    混合检索：向量语义 + BM25 关键词（实现在 rag_common.SimpleVectorStore.hybrid_search）。

    为什么加关键词：用户问「第一千零七十七条」「劳动合同法第三十九条」这类精确编号时，
    纯语义向量常常匹配不准，关键词却能一抓一个准，两者加权融合最稳。
    """
    store = get_store()
    docs = store.hybrid_search(query, k=k * 2, alpha=0.7)  # 多取一些，后面按阈值过滤

    scored = []
    for d in docs:
        score = d.metadata.get("score", 0.0)
        if score >= SCORE_THRESHOLD:
            scored.append((score, d))

    scored.sort(key=lambda x: -x[0])
    return scored[:k]


# ============================================================
# Agent 3：法律分析（作答）
# ============================================================
ANSWER_PROMPT = """你是中国法律顾问。请【只依据】下面检索到的法条作答，不得使用法条之外的知识。

检索到的法条：
{context}

用户问题：{question}

{extra}

要求：
1. 每个结论后面必须标注依据，格式如「（民法典第一千零七十七条）」。
2. 如果法条里没有答案，直接说"抱歉，当前知识库中未找到相关法条依据"，不要编造。
3. 语言通俗，但法律依据必须准确。

回答："""


def _build_answer_messages(question: str, docs: list, extra: str = ""):
    """构造 Agent 3 的完整消息（流式和非流式共用同一份提示词，保证行为一致）。"""
    from langchain_core.prompts import ChatPromptTemplate

    context = "\n\n".join(
        f"[{d.metadata.get('law_name', '')}] {d.page_content}" for _, d in docs
    ) or "（本次没有检索到任何法条）"
    return ChatPromptTemplate.from_template(ANSWER_PROMPT).format_messages(
        context=context, question=question, extra=extra
    )


def agent_answer(question: str, docs: list, extra: str = "") -> str:
    return call_llm(_build_answer_messages(question, docs, extra)).strip()


# ============================================================
# Agent 4：合规校验（查幻觉）
# ============================================================
VALIDATE_PROMPT = """你是严格的法律答案质检员。请逐句检查下面的回答，判断每一句是否都能在给定法条中找到依据。

法条：
{context}

用户问题：{question}

待检查的回答：
{answer}

判定规则（从严）：
1. 回答里每个结论都必须能对应到某条具体法条；只写"根据相关法律规定"却对不上具体条文的，判为无依据。
2. 法条里没有的补充推测、延伸建议、"需要注意/可能/一般来说"这类表述，一律判为无依据。
3. 引用了法条中不存在的内容（编造条文、张冠李戴），直接判 grounded=false。
4. 只有当回答是"未找到相关法条依据"、且法条里确实没有相关内容时，才算通过。

只输出一个 JSON 对象：
{{"grounded": true或false, "score": 0到1的小数, "issues": ["逐条列出无依据的原句，没有就留空数组"]}}
"""



def agent_validate(question: str, docs: list, answer: str) -> dict:
    context = "\n\n".join(d.page_content for _, d in docs) or "（无）"
    data = extract_json(
        call_llm(VALIDATE_PROMPT.format(context=context, question=question, answer=answer))
    )
    grounded = bool(data.get("grounded", True))
    try:
        score = float(data.get("score", 1.0))
    except (TypeError, ValueError):
        score = 1.0
    issues = data.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    return {"grounded": grounded, "score": score, "issues": issues}


# ============================================================
# 编排器：把 4 个 Agent 串起来
# ============================================================
def run_pipeline(question: str, k: int = 5, verbose: bool = True) -> dict:
    def log(msg):
        if verbose:
            print(msg)

    # ---- Agent 1 ----
    log("\n[Agent 1] 意图分析中...")
    intent = agent_intent(question)
    log(f"   领域：{intent['domain']}")
    log(f"   关键词：{', '.join(intent['keywords']) or '（无）'}")
    log(f"   改写后：{intent['rewritten_query']}")

    # ---- Agent 2 ----
    log("\n[Agent 2] 检索法条中...")
    docs = agent_retrieve(intent["rewritten_query"], intent["keywords"], k=k)
    if not docs:
        log("   没有检索到超过阈值的法条，将按'未找到'处理")
    for i, (score, d) in enumerate(docs, 1):
        log(f"   {i}. [{d.metadata.get('law_name','')}] {score:.3f}  {d.page_content[:40]}...")

    # ---- Agent 3 ----
    log("\n[Agent 3] 作答中...")
    answer = agent_answer(question, docs)
    log(f"   {answer[:80]}...")

    # ---- Agent 4 ----
    log("\n[Agent 4] 合规校验中...")
    check = agent_validate(question, docs, answer)
    log(f"   是否有依据：{'是' if check['grounded'] else '否'}  得分：{check['score']}")
    if check["issues"]:
        log(f"   问题：{check['issues']}")

    # ---- 不通过则让 Agent 3 重写（带校验意见）----
    retry = 0
    while not check["grounded"] and retry < MAX_RETRY and docs:
        retry += 1
        log(f"\n[重试 {retry}] 校验未通过，让 Agent 3 按意见重写...")
        extra = "上一次回答存在以下问题，请修正，删掉没有法条依据的表述：\n" + "\n".join(
            f"- {i}" for i in check["issues"]
        )
        answer = agent_answer(question, docs, extra=extra)
        check = agent_validate(question, docs, answer)
        log(f"   重写后得分：{check['score']}")

    return {
        "question": question,
        "intent": intent,
        "answer": answer,
        "grounded": check["grounded"],
        "score": check["score"],
        "issues": check["issues"],
        "retry": retry,
        "sources": [
            {
                "law_name": d.metadata.get("law_name", ""),
                "score": round(s, 4),
                "snippet": d.page_content,
            }
            for s, d in docs
        ],
    }


def _sources_payload(docs: list) -> list:
    return [
        {
            "law_name": d.metadata.get("law_name", ""),
            "article": d.metadata.get("article", ""),
            "score": round(s, 4),
            "snippet": d.page_content,
        }
        for s, d in docs
    ]


def run_pipeline_stream(question: str, k: int = 5):
    """
    流式版编排器：逻辑与 run_pipeline 完全一致，但把每一步中间结果 yield 出去，
    供 SSE 接口推给前端，用户能看到"检索到什么 → 逐字作答 → 校验结果"。

    事件类型：
      stage   阶段提示文案
      intent  意图分析结果
      sources 检索到的法条（先于答案返回，前端可提前渲染来源）
      token   答案的一个片段
      reset   校验未通过要重写，前端清空已渲染内容
      score   校验得分
      done    结束
    """
    yield {"type": "stage", "step": 0, "text": "意图分析"}
    intent = agent_intent(question)
    yield {"type": "intent", "domain": intent["domain"],
           "keywords": intent["keywords"], "rewritten": intent["rewritten_query"]}

    yield {"type": "stage", "step": 1, "text": "法条检索"}
    docs = agent_retrieve(intent["rewritten_query"], intent["keywords"], k=k)
    yield {"type": "sources", "sources": _sources_payload(docs),
           "keywords": intent["keywords"]}

    def stream_answer(extra: str = ""):
        """逐个 yield 文本片段（不负责累计，累计由调用方做）。"""
        for chunk in get_llm().stream(_build_answer_messages(question, docs, extra)):
            t = getattr(chunk, "content", "") or ""
            if t:
                yield t

    def emit_answer(extra: str = ""):
        """流式输出答案并同步累计完整文本，返回 (完整答案)。"""
        buf = []
        for t in stream_answer(extra):
            buf.append(t)
            yield {"type": "token", "text": t}
        return "".join(buf)

    yield {"type": "stage", "step": 2, "text": "依据作答"}
    answer = yield from emit_answer()

    yield {"type": "stage", "step": 3, "text": "合规校验"}
    check = agent_validate(question, docs, answer)

    retry = 0
    while not check["grounded"] and retry < MAX_RETRY and docs:
        retry += 1
        yield {"type": "stage", "step": 2, "text": f"校验未通过，重写（第 {retry} 次）"}
        yield {"type": "reset"}
        extra = "上一次回答存在以下问题，请修正，删掉没有法条依据的表述：\n" + "\n".join(
            f"- {i}" for i in check["issues"]
        )
        answer = yield from emit_answer(extra)
        check = agent_validate(question, docs, answer)

    yield {"type": "score", "score": check["score"], "grounded": check["grounded"],
           "issues": check["issues"], "retry": retry}
    yield {"type": "done"}

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "我借给朋友十万块，约定了利息，他两年不还，我还能起诉吗？"
    print("=" * 60)
    print(f"问题：{q}")
    print("=" * 60)
    result = run_pipeline(q)
    print("\n" + "=" * 60)
    print("最终答案：")
    print("=" * 60)
    print(result["answer"])
    print(f"\n依据法条数：{len(result['sources'])}  校验得分：{result['score']}  重写次数：{result['retry']}")
