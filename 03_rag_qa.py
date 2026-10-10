# 03_rag_qa.py —— 交互式法律问答
#
# 【原报错】Chroma(...) 缺 chromadb；ChatOpenAI 未显式传 api_key 时会读 OPENAI_API_KEY，
#           而本地实际配的是 DASHSCOPE_API_KEY，导致 401 / AuthenticationError。
# 【修复】  统一由 rag_common.get_api_key() 取 Key 并显式传入。
#
# 运行：python 03_rag_qa.py   （输入 exit 退出）
from rag_common import build_rag_chain, load_vector_store

print("=" * 60)
print("Step 3：RAG 法律问答")
print("=" * 60)

store = load_vector_store()
print(f"向量库已加载，共 {len(store)} 条法条片段。")

chain, retriever, _ = build_rag_chain(k=3, store=store)

print("\nRAG 系统启动成功！输入 'exit' 退出。")
print("提示：先看 [检索到的法条] 再回答，是判断 RAG 好不好用的关键。\n")

while True:
    try:
        query = input("你的法律问题: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not query:
        continue
    if query.lower() in ("exit", "quit", "q"):
        print("已退出。")
        break

    print("\n检索到的法条片段：")
    for i, d in enumerate(retriever.invoke(query), 1):
        src = d.metadata.get("law_name", "")
        print(f"  {i}. [{src}] {d.page_content[:80].replace(chr(10), ' ')}...")

    print("\n顾问思考中...")
    try:
        answer = chain.invoke(query)
        print(f"\n回答:\n{answer}")
    except Exception as e:
        print(f"调用大模型失败：{type(e).__name__}: {e}")
    print("-" * 60)
