# 02_vectorstore.py —— 向量化并持久化入库
#
# 【原报错1】ModuleNotFoundError: No module named 'chromadb'
# 【原因】    venv 里没装 chromadb，langchain_community.vectorstores.Chroma 只是薄封装。
# 【修复】    改用 rag_common.SimpleVectorStore（numpy + json 持久化，零额外依赖）。
#
# 【原报错2】HuggingFaceEmbeddings 报缺 sentence_transformers / torch
# 【原因】    venv 里没装 torch 全家桶，本地跑 bge-small-zh 需额外下载约 2GB。
# 【修复】    改用 DashScope text-embedding-v3 在线中文向量（1024 维，无需本地模型）。
#
# 运行：python 02_vectorstore.py
from rag_common import (
    STORE_DIR,
    build_embeddings,
    load_and_split,
    require_data_dir,
    SimpleVectorStore,
)

print("=" * 60)
print("Step 2：向量化与入库")
print("=" * 60)

require_data_dir()

chunks = load_and_split()
print(f"已加载并切片 {len(chunks)} 个文本块。")

print("正在调用 DashScope text-embedding-v3 生成中文向量（约需十几秒）...")
embeddings = build_embeddings()

store = SimpleVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=STORE_DIR,
    collection_name="chinese_law",
)
print(f"向量库已写入：{store.index_path.name} / {store.vector_path.name}")

print("\n验收测试：检索 '离婚冷静期是多少天？'")
for i, res in enumerate(store.similarity_search("离婚冷静期是多少天？", k=2), 1):
    src = res.metadata.get("law_name", "")
    score = res.metadata.get("score", 0)
    print(f"\n--- 匹配结果 {i}  [来源：{src}]  相似度：{score:.3f} ---")
    print(res.page_content[:200].replace("\n", " ") + "...")

print("\n下一步：运行 03_rag_qa.py 开始问答。")
