# 01_chunking.py —— 加载 data/ 下的法条并按 Markdown 标题层级切片
#
# 【原报错】ModuleNotFoundError: No module named 'langchain.text_splitter'
# 【原因】  langchain 1.x 重构后，主包不再导出 text_splitter，
#           MarkdownHeaderTextSplitter 已迁到独立包 langchain_text_splitters。
# 【修复】  统一走 rag_common.split_documents()（内部用新导入路径，并兼容新旧返回值）。
#
# 运行：python 01_chunking.py
from rag_common import load_and_split, require_data_dir

print("=" * 60)
print("Step 1：文档加载与切片")
print("=" * 60)

require_data_dir()

print("正在加载 data 目录下的文档...")
chunks = load_and_split()
print(f"切片完成！总共切出 {len(chunks)} 个文本块 (Chunks)。")

print("\n" + "=" * 60)
print("预览前 3 个文本块：")
print("=" * 60)
for i, chunk in enumerate(chunks[:3], 1):
    src = chunk.metadata.get("law_name", "")
    print(f"\n--- Chunk {i}  [来源：{src}]  长度：{len(chunk.page_content)} 字 ---")
    print(chunk.page_content[:200].replace("\n", " ") + "...")

# 简单体检：块太大会拖垮检索，块太小会丢失语义
sizes = sorted(len(c.page_content) for c in chunks)
if sizes:
    print("\n" + "-" * 60)
    print(f"文本块长度分布：最小 {sizes[0]} 字 / 中位数 {sizes[len(sizes)//2]} 字 / 最大 {sizes[-1]} 字")
    too_long = [c for c in chunks if len(c.page_content) > 1500]
    if too_long:
        print(f"提示：有 {len(too_long)} 个块超过 1500 字，检索精度会下降，建议按「条」再细分。")
print("\n下一步：运行 02_vectorstore.py 构建向量库。")
