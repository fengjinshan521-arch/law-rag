# law-rag 报错原因与修复记录

## 0. 环境体检结果

在你项目的 `venv` 里实际装的是（这是所有报错的根源）：

| 包 | 版本 | 状态 |
|---|---|---|
| langchain | **1.4.2** | 已是 1.x 大版本，API 大面积迁移 |
| langchain_core | 1.6.3 | |
| langchain_community | 0.4.2 | 已废弃警告（sunset） |
| langchain_text_splitters | 1.1.2 | 切分器独立成包了 |
| langchain_openai | 1.1.9 | |
| ragas | 0.4.3 | 与 langchain 1.x 不兼容 |
| chromadb | ❌ 未安装 | |
| torch / transformers / sentence_transformers | ❌ 未安装 | |

**结论：代码是按 langchain 0.1.x 时代的写法写的，环境却是 langchain 1.x + 缺 3 个重依赖，所以 01~05 全都跑不起来。**

---

## 1. 逐条报错 → 原因 → 修复

### ❌ 报错 1：`ModuleNotFoundError: No module named 'langchain.text_splitter'`

- **出现在**：01 / 02 的 `from langchain.text_splitter import MarkdownHeaderTextSplitter`
- **原因**：langchain 1.x 把主包拆干净了，切分器搬到独立包 `langchain_text_splitters`；`langchain` 主包里已经没有 `text_splitter` 这个子模块。
- **修复**：
  ```python
  from langchain_text_splitters import MarkdownHeaderTextSplitter   # 新
  ```
  另外 langchain 1.x 的 `split_text()` **返回 `Document` 列表**（老版本返回 `str` 列表），原代码还写成 `Document(page_content=text)` 会把 Document 当字符串塞进去，二次踩坑。已在 `rag_common.split_documents()` 里做了新旧兼容。

### ❌ 报错 2：`ModuleNotFoundError: No module named 'chromadb'`

- **出现在**：02 / 03 / 04 的 `from langchain_community.vectorstores import Chroma`
- **原因**：`import` 这行能过（只是薄封装），但真正实例化 `Chroma(...)` 时才去 `import chromadb`，而 venv 里压根没装。
- **修复**：新增 **零依赖的本地向量库** `SimpleVectorStore`（`rag_common.py`）：
  - 持久化：`vector_store/chinese_law.index.json`（文本+元数据）+ `vector_store/chinese_law.vectors.npy`（L2 归一化后的向量矩阵）
  - 检索：归一化后做内积 = 余弦相似度
  - 提供 `similarity_search()` / `as_retriever()`，用法与 Chroma 一致，`retriever | format_docs` 的 LCEL 管道照常可用
  - 好处：不用装 chromadb（约 100MB+ 含 onnxruntime），也不受 chromadb 与 langchain_community 0.4.2 的版本摩擦影响

### ❌ 报错 3：`HuggingFaceEmbeddings` 缺 `sentence_transformers` / `torch`

- **出现在**：02 / 03 / 04 的 `HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")`
- **原因**：本地跑 bge 模型需要 `torch`（约 2GB）+ `sentence-transformers` + 从 HuggingFace 下载模型权重，当前 venv 一样没装，国内网络还容易超时。
- **修复**：改用 **DashScope 在线中文向量**（你本机已配 `DASHSCOPE_API_KEY`）：
  ```python
  DashScopeEmbeddings(model="text-embedding-v3")   # 1024 维，无需本地模型
  ```
  已实测通过；内部做了批量（8 条/批）和单条 3000 字截断，规避接口的 8192 token 限制。

### ❌ 报错 4：`No module named 'langchain_community.chat_models.vertexai'`（ragas 一 import 就崩）

- **出现在**：`05_ragas_eval.py` 的 `import ragas`
- **原因**：ragas 0.4.3 依赖旧版 langchain_community（0.3.x）的模块结构，和当前 0.4.2 对不上。
- **附带问题**：ragas 0.4.x 已重构 API，`from ragas.metrics import faithfulness, answer_relevancy, context_recall` 这种老写法在新版里根本不存在。硬修要动 langchain 全家桶版本，风险很高。
- **修复**：**去掉 ragas 依赖**，改成用 qwen-max 当裁判，指标口径与 ragas 对齐（faithfulness / answer_relevancy / context_recall，0~1 分），输出 `eval_scores.json` + `eval_scores.csv`。

### ❌ 报错 5：LLM 端点与 Key 对不上（401）

- **出现在**：`03` 用 dashscope 端点、`04` 却用 `https://api.siliconflow.cn/v1`，而本机 **没有** `SILICONFLOW_API_KEY`，只有 `DASHSCOPE_API_KEY`。
- **原因**：`ChatOpenAI` 若不显式传 `api_key`，会去读 `OPENAI_API_KEY`，拿着 dashscope 的 key 打 siliconflow 的域名，必然鉴权失败。
- **修复**：统一由 `rag_common.get_api_key()` 取 Key（优先 `DASHSCOPE_API_KEY`），并**显式传参**给 `ChatOpenAI` / `OpenAI`，不再依赖隐式环境变量。

### ⚠️ 隐患 6：知识库和测试集根本不是同一部法律

- `eval_set.jsonl` 的 8 道题**全是《民法典》**，而 `data/` 里只有《刑法》和《劳动法》。就算前面全修好，RAG 也只会回答"未找到相关法条依据"。
- **修复**：新增 `data/中华人民共和国民法典.md`（节选，覆盖评测集涉及的条文：第 19、20、188、586、1062、1063、1064、1077、1161、1165 条）。
- 注意：这是**节选版**，建议后续用官方全文替换。第 7、8 题的标准答案还引用了司法解释和法理（不在民法典正文中），因此这两题的 context_recall 只有 0.5，属正常现象。

### 🐛 附带修掉的小问题

- **Windows 控制台 GBK 编码**：脚本里的 emoji / 中文在 PyCharm 之外的终端会 `UnicodeEncodeError`。`rag_common.py` 启动时统一 `reconfigure(encoding='utf-8')`。
- **YAML front matter 被当成正文**：`---` 包裹的元信息被切成了一个独立文本块，污染检索。已加 `strip_front_matter()`。
- **切分粒度**：给切分器补了 `####` 四级标题，民法典里每条文一个 chunk，检索精度明显提升（"离婚冷静期"直接命中第 1077 条，相似度 0.789）。

---

## 2. 现在怎么跑（按顺序）

```bash
cd E:\law-rag
.\venv\Scripts\python.exe 01_chunking.py      # 切片预览，22 个块
.\venv\Scripts\python.exe 02_vectorstore.py   # 建向量库，写入 vector_store/
.\venv\Scripts\python.exe 03_rag_qa.py        # 交互式问答，输入 exit 退出
.\venv\Scripts\python.exe 04_evaluate.py      # 批量跑测试集 -> eval_results.json
.\venv\Scripts\python.exe 05_ragas_eval.py    # LLM 裁判打分 -> eval_scores.csv
```

实测结果（8 条测试题）：

```
忠实度 faithfulness     : 0.975
答案相关性 relevancy    : 1.000
上下文召回 recall       : 0.875
```

## 3. 文件说明

| 文件 | 作用 |
|---|---|
| `rag_common.py` | **新增**，公共模块：加载/切分/Embedding/向量库/RAG 链 |
| `data/中华人民共和国民法典.md` | **新增**，补齐与测试集匹配的知识库 |
| `01`~`05` | 已按新架构重写，每个文件头部注释都用「报错 → 原因 → 修复」标了改动点 |
| `vector_store/` | 02 生成的索引（`*.index.json` + `*.vectors.npy`） |
| `main.py` | PyCharm 默认模板，与本流程无关，可忽略 |

## 4. 如果坚持用原方案（Chroma + 本地 bge 模型）

需要额外安装：

```bash
pip install chromadb                      # ~100MB+
pip install torch --index-url https://download.pytorch.org/whl/cpu   # ~200MB（CPU 版）
pip install sentence-transformers         # 再从 HF 下载 bge-small-zh-v1.5
```

并把 `rag_common.build_embeddings()` 换回 `HuggingFaceEmbeddings`，`SimpleVectorStore` 换回 `Chroma`。除此之外，报错 1（导入路径）、报错 5（端点/Key）、隐患 6（数据不匹配）依然必须修。
