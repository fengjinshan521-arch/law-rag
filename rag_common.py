# rag_common.py
"""
law-rag 公共模块：把 01~05 脚本共用的能力集中在这里，避免每个脚本各写一遍。

职责：
  1. 加载 data/ 下的 Markdown 法律文本
  2. 按 Markdown 标题层级切块（兼容 langchain 1.x 的新导入路径）
  3. 调用 DashScope（OpenAI 兼容模式）的在线中文向量模型
  4. 提供一个零额外依赖的本地向量库（numpy 持久化，替代 chromadb）
  5. 组装 RAG 问答链

修复说明见 README_FIX.md
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from langchain_core.embeddings import Embeddings

# ---------- 控制台编码：Windows 默认 GBK，打印 emoji/中文会炸 ----------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORE_DIR = BASE_DIR / "vector_store"
EVAL_SET = BASE_DIR / "eval_set.jsonl"

# ---------- 读取 .env（可选）与 API Key ----------
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-v3")   # 1024 维中文向量
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-max")


def get_api_key() -> str:
    """优先用 DASHSCOPE_API_KEY，其次 OPENAI_API_KEY。"""
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "未找到 API Key。请在系统环境变量或项目根目录的 .env 中配置：\n"
            "  DASHSCOPE_API_KEY=sk-xxxxxxxx\n"
            "获取地址：https://bailian.console.aliyun.com/"
        )
    return key


def require_data_dir() -> None:
    if not DATA_DIR.exists() or not any(DATA_DIR.rglob("*.md")):
        raise RuntimeError(f"data/ 目录下没有找到 .md 文件，请先把法条放进 {DATA_DIR}")


# ============================================================
# 1. 文档加载 + 切分
# ============================================================
# 中文条号，如「第一千零七十七条」「第一百八十八条」「第十九条」
ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千零〇\d]+条")


def strip_front_matter(text: str) -> str:
    """去掉 Markdown 顶部的 YAML front matter（--- ... ---），否则它会被当成正文切成一块。"""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + 4 :].lstrip("\n")


def load_documents(data_dir: Path = DATA_DIR):
    """加载目录下所有 .md 文件，返回一个个 Document。"""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_core.documents import Document

    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()

    # DirectoryLoader 在部分版本下不回填 source，这里兜底补一次
    for d in docs:
        src = d.metadata.get("source")
        if not src:
            d.metadata["source"] = str(data_dir)
        d.metadata["law_name"] = Path(d.metadata["source"]).stem
        d.page_content = strip_front_matter(d.page_content)
    return docs


def _prev_heading(text: str, pos: int) -> str:
    """找出 pos 之前最近的一个有意义标题，作为该条文的所属章节。

    跳过两类干扰：空标题（如条号所在行 "#### 第一千..." 被截断后只剩 "####"），
    以及条号本身就是标题的情况（如 "#### 第一千零七十七条"）。
    """
    head = text[:pos]
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()]
    for ln in reversed(lines[-40:]):        # 只往回看 40 行，够用且快
        if not ln.startswith("#"):
            continue
        title = ln.lstrip("#").strip()
        if title and not re.match(r"^第[一二三四五六七八九十百千零〇\d]+条", title):
            return title
    return ""


def split_by_article(docs) -> List["Document"]:  # noqa: F821
    """
    按「第X条」切分——法律文本最合理的粒度。

    为什么不用 Markdown 标题切：刑法/劳动法的数据里，十几条法条挤在同一个标题下
    连成一大块（最大的一块 1323 字），检索时只能整块返回，精度很差。
    按条切开后，每条独立成块，还能把条号写进元数据，回答时就能精确引用。

    条号误判防护：只认「第X条」前面是行首/空白/句号等位置的情况，
    避免把正文里"依照本法第一千零六十三条"这种引用当成新条文的开头。
    """
    from langchain_core.documents import Document

    article_re = re.compile(r"第[一二三四五六七八九十百千零〇\d]+条")
    chunks: List[Document] = []

    def is_start(pos: int, text: str) -> bool:
        if pos == 0:
            return True
        prev = text[pos - 1]
        return prev.isspace() or prev in "。.;；:：、（(（"

    for doc in docs:
        text = doc.page_content
        source = doc.metadata.get("source", "")
        law = doc.metadata.get("law_name", "")
        matches = [m for m in article_re.finditer(text) if is_start(m.start(), text)]
        if not matches:
            continue                      # 这篇文档没有条号，交给 Markdown 切分兜底

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            # 截断：遇到下一个标题行就停，避免把下一编/下一章的标题混进本条
            seg = text[start:end]
            lines = seg.split("\n")
            kept = [lines[0]]
            for ln in lines[1:]:
                if ln.strip().startswith("#"):
                    break
                kept.append(ln)
            seg = "\n".join(kept).strip()
            if len(seg) < 15:             # 太短的碎片丢弃
                continue
            article = m.group(0)
            chapter = _prev_heading(text, start)
            content = f"{law} {article}\n"
            if chapter:
                content += f"所属：{chapter}\n"
            content += seg
            chunks.append(Document(
                page_content=content,
                metadata={"source": source, "law_name": law,
                          "article": article, "chapter": chapter},
            ))
    return chunks


def split_documents(docs) -> List["Document"]:  # noqa: F821
    """
    先按「第X条」切（法律文本的最佳粒度），切不出条号的文档再退回 Markdown 标题切分。
    """
    article_chunks = split_by_article(docs)
    if article_chunks:
        return article_chunks
    return _split_by_markdown(docs)


def _split_by_markdown(docs) -> List["Document"]:  # noqa: F821
    """
    按 Markdown 标题层级切块（兜底方案）。
    注意：langchain 1.x 之后 MarkdownHeaderTextSplitter 已迁到
    langchain_text_splitters，且 split_text 返回的是 Document 列表（旧版返回 str）。
    这里对两种返回都做兼容。
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),   # 民法典数据用四级标题把「每一条」切成独立块，检索更精准
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    chunks: List[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        try:
            parts = splitter.split_text(doc.page_content)
        except Exception:
            parts = []

        if not parts:  # 文档里没有 Markdown 标题时兜底：按段落粗切
            raw = [p.strip() for p in doc.page_content.split("\n\n") if p.strip()]
            parts = [Document(page_content=p) for p in raw] if raw else [doc]

        for p in parts:
            if isinstance(p, str):
                p = Document(page_content=p)
            text = (p.page_content or "").strip()
            if len(text) < 10:      # 过滤只剩标题的空块
                continue
            meta = dict(p.metadata or {})
            meta.setdefault("source", source)
            meta.setdefault("law_name", Path(source).stem if source else "")
            chunks.append(Document(page_content=text, metadata=meta))
    return chunks


def load_and_split(data_dir: Path = DATA_DIR) -> List["Document"]:  # noqa: F821
    return split_documents(load_documents(data_dir))


# ============================================================
# 2. DashScope 在线中文 Embedding
# ============================================================
class DashScopeEmbeddings(Embeddings):
    """DashScope text-embedding-v3，OpenAI 兼容接口。无需 torch / sentence-transformers。"""

    MAX_CHARS = 3000      # 单条文本截断上限，规避 8192 token 限制
    BATCH_SIZE = 8        # 单请求条数上限（官方 ≤10）

    def __init__(self, model: str = EMBED_MODEL, api_key: Optional[str] = None):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key or get_api_key(), base_url=DASHSCOPE_BASE_URL)

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        cleaned = []
        for t in texts:
            t = (t or "").replace("\n", " ").strip()
            cleaned.append(t[: self.MAX_CHARS] if t else " ")

        vecs: List[List[float]] = []
        for i in range(0, len(cleaned), self.BATCH_SIZE):
            resp = self._client.embeddings.create(model=self.model, input=cleaned[i : i + self.BATCH_SIZE])
            vecs.extend([d.embedding for d in resp.data])
        return vecs

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def build_embeddings() -> DashScopeEmbeddings:
    return DashScopeEmbeddings()


class KeywordIndex:
    """
    BM25 关键词索引。补向量检索的短板：
    用户问「第一千零七十七条」「劳动合同法第三十九条」这种精确编号时，
    纯语义向量常常匹配不准，而关键词匹配一抓一个准。
    """

    K1, B = 1.5, 0.75

    def __init__(self):
        self.docs_tokens: List[List[str]] = []
        self.df: dict = {}
        self.lens: List[int] = []
        self.avgdl = 1.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        import jieba

        text = (text or "").lower()
        return [w for w in jieba.lcut(text) if len(w) > 1 and w.strip()]

    def build(self, texts: Sequence[str]) -> None:
        self.docs_tokens = [self.tokenize(t) for t in texts]
        self.build_docs_tokens()

    def build_docs_tokens(self) -> None:
        """由已有的 docs_tokens 重新计算统计量（加载索引文件后调用）。"""
        self.lens = [len(t) for t in self.docs_tokens]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df: dict = {}
        for toks in self.docs_tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        self.df = df

    def scores(self, query: str) -> List[float]:
        import math

        q = self.tokenize(query)
        n = len(self.docs_tokens)
        out = []
        for i, toks in enumerate(self.docs_tokens):
            s = 0.0
            dl = self.lens[i] or 1
            for w in q:
                df = self.df.get(w)
                if not df:
                    continue
                tf = toks.count(w)
                if not tf:
                    continue
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * tf * (self.K1 + 1) / (
                    tf + self.K1 * (1 - self.B + self.B * dl / self.avgdl)
                )
            out.append(s)
        return out


# ============================================================
# 3. 轻量本地向量库（numpy + json，替代 chromadb）
# ============================================================
class SimpleVectorStore:
    """
    极简持久化向量库：
      vector_store/<collection>/index.json   -> texts + metadatas
      vector_store/<collection>/vectors.npy  -> 已 L2 归一化的向量矩阵
      vector_store/<collection>/keyword.json -> BM25 关键词索引
    支持两种检索：similarity_search（纯向量）与 hybrid_search（向量 + BM25）。
    """

    def __init__(self, persist_directory: Path, collection_name: str = "chinese_law",
                 embedding_function=None):
        self.dir = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or build_embeddings()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / f"{collection_name}.index.json"
        self.vector_path = self.dir / f"{collection_name}.vectors.npy"
        self.keyword_path = self.dir / f"{collection_name}.keyword.json"
        self.texts: List[str] = []
        self.metadatas: List[dict] = []
        self._vectors = None  # numpy 2D array，懒加载
        self.keyword = KeywordIndex()

    # ---------- 写 ----------
    @classmethod
    def from_documents(cls, documents, embedding=None, persist_directory=STORE_DIR,
                       collection_name="chinese_law") -> "SimpleVectorStore":
        store = cls(persist_directory, collection_name, embedding)
        store.add_documents(documents)
        store.persist()
        return store

    def add_documents(self, documents) -> None:
        texts = [d.page_content for d in documents]
        metas = [dict(d.metadata or {}) for d in documents]
        vecs = self.embedding_function.embed_documents(texts)
        self.texts = texts
        self.metadatas = metas
        self._vectors = self._as_matrix(vecs)
        self.keyword.build(texts)

    def persist(self) -> None:
        import numpy as np

        if self._vectors is None:
            raise RuntimeError("没有可持久化的向量，请先 add_documents()")
        np.save(self.vector_path, self._vectors)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(
                {"collection": self.collection_name, "count": len(self.texts),
                 "texts": self.texts, "metadatas": self.metadatas},
                f, ensure_ascii=False,
            )
        with open(self.keyword_path, "w", encoding="utf-8") as f:
            json.dump({"docs_tokens": self.keyword.docs_tokens}, f, ensure_ascii=False)

    # ---------- 读 ----------
    def _load(self) -> None:
        import numpy as np

        if self._vectors is not None:
            return
        if not (self.index_path.exists() and self.vector_path.exists()):
            raise FileNotFoundError(
                f"向量库不存在：{self.index_path}\n请先运行 02_vectorstore.py 构建索引。"
            )
        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = data["texts"]
        self.metadatas = data.get("metadatas", [{} for _ in self.texts])
        self._vectors = np.load(self.vector_path)
        # 关键词索引是可选的：老版本索引没有它时自动重建，保证不报错
        if self.keyword_path.exists():
            with open(self.keyword_path, "r", encoding="utf-8") as f:
                self.keyword.docs_tokens = json.load(f).get("docs_tokens", [])
            self.keyword.build_docs_tokens()
        else:
            self.keyword.build(self.texts)

    @staticmethod
    def _as_matrix(vecs) -> "np.ndarray":  # noqa: F821
        import numpy as np

        m = np.asarray(vecs, dtype="float32")
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return m / norm

    def __len__(self) -> int:
        self._load()
        return len(self.texts)

    def similarity_search(self, query: str, k: int = 3):
        """返回与 query 最相似的 k 个 Document。"""
        import numpy as np
        from langchain_core.documents import Document

        self._load()
        q = self._as_matrix([self.embedding_function.embed_query(query)])[0]
        scores = self._vectors @ q
        k = min(k, len(self.texts))
        top = np.argsort(-scores)[:k]
        out = []
        for idx in top:
            out.append(
                Document(
                    page_content=self.texts[int(idx)],
                    metadata={**self.metadatas[int(idx)], "score": float(scores[int(idx)])},
                )
            )
        return out

    def hybrid_search(self, query: str, k: int = 3, alpha: float = 0.7):
        """
        混合检索：向量语义分（权重 alpha） + BM25 关键词分（权重 1-alpha）。

        alpha=0.7 表示更信任语义；遇到「第1077条」这种精确编号时，
        BM25 会把它捞上来，弥补纯向量的不足。
        """
        import numpy as np
        from langchain_core.documents import Document

        self._load()
        q = self._as_matrix([self.embedding_function.embed_query(query)])[0]
        vec_scores = self._vectors @ q
        kw_scores = np.asarray(self.keyword.scores(query), dtype="float32")

        def norm(a):
            if a.size == 0:
                return a
            lo, hi = float(a.min()), float(a.max())
            return (a - lo) / (hi - lo) if hi > lo else a * 0.0

        combined = alpha * norm(vec_scores) + (1 - alpha) * norm(kw_scores)

        # 条号精确匹配：用户直接问「第一千零七十七条」时，jieba 会把条号切碎导致 BM25 失效，
        # 所以这里用整串匹配单独加一个强权重，确保精确编号查询能命中。
        arts = ARTICLE_PATTERN.findall(query)
        if arts:
            import numpy as _np
            bonus = _np.zeros_like(combined)
            for i, t in enumerate(self.texts):
                if any(a in t for a in arts):
                    bonus[i] = 1.0
            combined = combined + bonus
        k = min(k, len(self.texts))
        top = np.argsort(-combined)[:k]
        out = []
        for idx in top:
            out.append(
                Document(
                    page_content=self.texts[int(idx)],
                    metadata={
                        **self.metadatas[int(idx)],
                        "score": float(combined[int(idx)]),
                        "vec_score": float(vec_scores[int(idx)]),
                        "kw_score": float(kw_scores[int(idx)]),
                    },
                )
            )
        return out

    def as_retriever(self, k: int = 3):
        return SimpleRetriever(store=self, k=k)


class SimpleRetriever:
    """
    实现 LangChain Retriever 协议，可直接用 `retriever | format_docs`
    或 `retriever.invoke("问题")`，用法和 Chroma 的 as_retriever() 一致。
    """

    def __init__(self, store: SimpleVectorStore, k: int = 3):
        self.store = store
        self.k = k

    def invoke(self, query: str, **kwargs):
        return self.store.similarity_search(query, k=self.k)

    def get_relevant_documents(self, query: str):
        return self.invoke(query)

    def __or__(self, other):
        """支持 LCEL 的 `retriever | format_docs` 管道写法。"""
        from langchain_core.runnables import RunnableLambda
        from langchain_core.runnables.base import Runnable

        left = RunnableLambda(self.invoke)
        if isinstance(other, Runnable):
            return left | other
        return left | RunnableLambda(other)


def load_vector_store(persist_directory=STORE_DIR, collection_name="chinese_law") -> SimpleVectorStore:
    return SimpleVectorStore(persist_directory, collection_name)


# ============================================================
# 4. RAG 链
# ============================================================
RAG_PROMPT_TEMPLATE = """你是一个专业的中国法律顾问。请【严格仅根据】以下检索到的法律条文片段来回答用户的问题。
如果检索到的片段中没有相关信息，请直接回答“抱歉，当前知识库中未找到相关法条依据”，绝对不要自己编造。
回答时请注明依据的法律名称和条文序号。

检索到的法条片段：
{context}

用户问题：{question}

专业回答："""


def build_llm(model: str = LLM_MODEL, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=DASHSCOPE_BASE_URL,
        api_key=get_api_key(),
        temperature=temperature,
        timeout=120,
        max_retries=2,
    )


def format_docs(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("law_name", "")
        tag = f"[来源：{src}]" if src else ""
        parts.append(f"片段{i} {tag}\n{d.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(k: int = 3, store: Optional[SimpleVectorStore] = None, llm=None):
    """返回 (rag_chain, retriever, store)。"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    store = store or load_vector_store()
    retriever = store.as_retriever(k=k)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm = llm or build_llm()

    chain = (
        {
            "context": RunnableLambda(lambda q: retriever.invoke(q)) | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever, store
