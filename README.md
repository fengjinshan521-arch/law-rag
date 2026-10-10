# 法律领域 RAG 问答系统

基于 LangChain + 通义千问（DashScope）构建的法律领域检索增强生成（RAG）问答系统，针对法条文本「条款密集、结构规整」的特点做了定向优化，并配套完整的自动化评测工作流。

## 核心特性

- **四 Agent 协作流水线**：意图改写 → 混合检索 → 依据作答 → 合规校验，由合规校验 Agent 对答案中的引用与表述逐条核查，校验不通过时自动重写并回流修正，形成闭环。
- **条粒度文档切分**：按「条」粒度切分法条并挂载条号元数据，最大文本块由 1323 字降至 293 字，降低跨条款混淆带来的召回噪声。
- **混合检索**：向量检索 + BM25 混合召回，并针对法条编号做精确匹配优化，修正纯向量检索场景下的条号错配问题。
- **自动化评测**：以 36 题测试集输出法条命中率 / MRR / 幻觉分（依据分）三项指标，支持多轮迭代对比，持续驱动检索与提示策略优化。
- **服务化交付**：基于 FastAPI 提供 SSE 流式输出与答案来源高亮溯源，支持法条文件上传并自动重建索引。

## 评测结果

| 指标 | 结果 |
|---|---|
| 法条命中率 | 100% |
| MRR 排序质量 | 0.95 |
| 有依据回答占比 | 100% |

逐题明细与多轮对比见 `reports/`，评测数据见 `eval_results.json`、`eval_scores.json`。

## 项目结构

```
law-rag/
├── 01_chunking.py          # 法条按「条」粒度切分
├── 02_vectorstore.py       # 向量库构建（DashScope 中文向量）
├── 03_rag_qa.py            # RAG 问答主链路
├── 04_evaluate.py          # 命中率 / MRR / 依据分自动化评测
├── 05_ragas_eval.py        # RAGAS 指标评测（忠实度/相关性/召回）
├── 06_multi_agent.py       # 四 Agent 协作流水线
├── 07_auto_workflow.py     # 评测-重写-回流 自动化工作流
├── api_server.py           # FastAPI 服务（SSE 流式 + 来源溯源）
├── rag_common.py           # 公共配置（模型接入、检索封装）
├── multi_agent_runner.py   # 多 Agent 运行入口
├── data/                   # 法条语料（民法典 / 刑法 / 劳动法）
├── reports/                # 评测报告
└── eval_*.json / *.jsonl   # 评测集与结果
```

## 快速开始

```bash
# 1. 安装依赖
python -m venv venv
venv\Scripts\pip install -r requirements.txt   # 或按各脚本头部注释安装

# 2. 配置 API Key（环境变量）
set DASHSCOPE_API_KEY=sk-xxxxxxxx

# 3. 构建向量库
python 02_vectorstore.py

# 4. 启动问答 / 评测
python 03_rag_qa.py
python 04_evaluate.py

# 5. 启动 API 服务
uvicorn api_server:app --reload
```

## 技术栈

Python · LangChain · 通义千问（DashScope 兼容 OpenAI 接口）· 向量检索 + BM25 混合召回 · 多 Agent 编排 · RAGAS 评测 · FastAPI（SSE）
