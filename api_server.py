# api_server.py —— 用 FastAPI 把整套法律 RAG 发布成服务
#
# 【为什么不用 05_ragas_eval.py 跑 uvicorn】
#   uvicorn 05_ragas_eval:app  -> ERROR: Attribute "app" not found in module "05_ragas_eval"
#   05 是批处理脚本（跑完就退出），没有 app 对象；且模块名以数字开头不是合法标识符。
#
# 【启动】
#   uvicorn api_server:app --reload --host 0.0.0.0 --port 8080
#   浏览器打开 http://localhost:8080 直接用问答页面，接口文档在 /docs
#
# 【接口】
#   GET  /               问答网页（HTML）
#   GET  /api/stats      知识库统计（条数、法律文件列表）
#   POST /api/ask        问答，返回答案 + 命中的法条来源
#   POST /api/ask/stream SSE 流式问答（逐字输出 + 中间步骤）
#   POST /api/rebuild    重新切片并重建向量库
#   GET  /api/files      列出 data/ 下的法条文件
#   POST /api/upload     上传法条文件到 data/（.md / .txt）
#   DELETE /api/files/{name} 删除某个法条文件
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_common import (
    DATA_DIR,
    STORE_DIR,
    build_embeddings,
    build_rag_chain,
    load_and_split,
    load_vector_store,
    require_data_dir,
    SimpleVectorStore,
)

MAX_UPLOAD_MB = 10      # 单个上传文件大小上限

_state: dict = {}


def _refresh() -> int:
    """（重新）加载向量库与 RAG 链，返回知识库条数。"""
    store = load_vector_store()
    chain, retriever, store = build_rag_chain(k=3, store=store)
    _state["chain"] = chain
    _state["retriever"] = retriever
    _state["size"] = len(store)
    return _state["size"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    require_data_dir()
    n = _refresh()
    print(f"RAG 服务就绪：知识库共 {n} 条法条片段")
    yield
    _state.clear()


app = FastAPI(title="law-rag 法律问答服务", version="1.0", lifespan=lifespan)


# ---------------- 数据模型 ----------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="法律问题")
    top_k: int = Field(3, ge=1, le=10, description="检索返回的法条片段数")


# ---------------- 页面 ----------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>法律 RAG 问答</title>
<style>
:root{--bg:#f6f7fb;--card:#fff;--ink:#141c26;--muted:#6b7280;--line:#e6e9ef;--brand:#2547d0;--brand-soft:#eef2ff;--ok:#0f7b5f;--warn:#b45309;--mark:#fde68a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:880px;margin:0 auto;padding:36px 20px 72px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:22px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:40px;height:40px;border-radius:12px;background:#2547d0;display:flex;align-items:center;justify-content:center;color:#fff;font-size:18px}
h1{font-size:20px;margin:0;font-weight:600}
.sub{margin:4px 0 0;color:var(--muted);font-size:13px}
.pill{display:inline-flex;align-items:center;gap:7px;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:7px 13px;font-size:12px;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px}
textarea{width:100%;min-height:88px;padding:14px 16px;font-size:15px;font-family:inherit;border:1px solid var(--line);border-radius:11px;resize:vertical;outline:none;line-height:1.6}
textarea:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(37,71,208,.12)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.chip{background:var(--brand-soft);color:var(--brand);border:none;border-radius:999px;padding:6px 12px;font-size:12.5px;cursor:pointer;font-family:inherit}
.chip:hover{background:#dfe6ff}
.row{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:14px}
button.primary{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:11px 26px;font-size:15px;font-family:inherit;cursor:pointer;font-weight:500}
button.primary:hover{background:#1c3bb0}
button.primary:disabled{background:#a9b6d9;cursor:not-allowed}
.switch{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--muted);cursor:pointer;user-select:none}
select{padding:7px 10px;border-radius:8px;border:1px solid var(--line);font-family:inherit;font-size:13px;background:#fff;color:var(--ink)}
.spacer{flex:1}
.steps{display:flex;gap:8px;margin-bottom:16px}
.step{flex:1;background:#f2f4f8;border-radius:10px;padding:9px 11px;font-size:12px;color:var(--muted);display:flex;align-items:center;gap:7px}
.step .n{width:18px;height:18px;border-radius:50%;background:#dfe3ea;color:#6b7280;font-size:11px;display:flex;align-items:center;justify-content:center;flex:none}
.step.on{background:var(--brand-soft);color:var(--brand)}
.step.on .n{background:var(--brand);color:#fff}
.step.done{color:var(--ok)}
.step.done .n{background:var(--ok);color:#fff}
.sec{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}
.sec h2{font-size:14px;margin:0;font-weight:600}
.meta{font-size:12px;color:var(--muted)}
.answer{white-space:pre-wrap;line-height:1.85;font-size:15px}
.caret{display:inline-block;width:2px;height:16px;background:var(--brand);vertical-align:-3px;margin-left:2px;animation:blink 1s steps(2,start) infinite}
@keyframes blink{to{visibility:hidden}}
mark{background:var(--mark);color:#141c26;padding:0 2px;border-radius:3px}
.cite{color:var(--brand);cursor:pointer;border-bottom:1px dashed var(--brand);padding:0 1px}
.cite:hover{background:var(--brand-soft);border-radius:3px}
.src{border:1px solid var(--line);border-radius:11px;padding:12px 14px;margin-bottom:10px;background:#fcfcfd}
.src.hl{border-color:#f0b429;background:#fffbeb}
.src .top{display:flex;align-items:center;gap:8px;margin-bottom:7px;flex-wrap:wrap}
.tag{font-size:11.5px;padding:2px 8px;border-radius:6px;background:#eef1f6;color:#4b5563}
.tag.law{background:#e8f0ff;color:#2547d0}
.tag.art{background:#e7f6ef;color:#0f7b5f}
.bar{flex:1;height:5px;background:#eef0f4;border-radius:3px;overflow:hidden;min-width:56px}
.bar i{display:block;height:100%;background:#2547d0}
.src .text{font-size:13.5px;color:#3d4753;line-height:1.75;white-space:pre-wrap}
.mini{border:none;background:transparent;color:var(--muted);font-size:12px;cursor:pointer;font-family:inherit;padding:4px 8px;border-radius:6px}
.mini:hover{background:#f0f2f6;color:var(--ink)}
.empty{color:#9aa3af;font-size:13.5px}
.err{color:#b91c1c;font-size:14px}
footer{text-align:center;color:#9aa3af;font-size:12px;margin-top:26px}

.drop{border:1.5px dashed #c9d2e3;border-radius:12px;padding:22px;text-align:center;color:var(--muted);font-size:13.5px;cursor:pointer;background:#fbfcfe;transition:.15s}
.drop:hover{border-color:var(--brand);background:var(--brand-soft);color:var(--brand)}
.drop.over{border-color:var(--brand);background:var(--brand-soft);color:var(--brand)}
.flist{margin-top:12px;display:flex;flex-direction:column;gap:8px}
.fitem{display:flex;align-items:center;gap:10px;background:#fcfcfd;border:1px solid var(--line);border-radius:10px;padding:9px 12px;font-size:13px}
.fitem .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fitem .sz{color:var(--muted);font-size:12px}
.del{border:none;background:transparent;color:#c2c8d2;cursor:pointer;font-size:15px;line-height:1;padding:2px 6px;border-radius:6px}
.del:hover{background:#fee2e2;color:#b91c1c}
.kbrow{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
button.ghost{background:#fff;color:var(--brand);border:1px solid #c9d2e3;border-radius:9px;padding:8px 16px;font-size:13.5px;font-family:inherit;cursor:pointer}
button.ghost:hover{background:var(--brand-soft)}
button.ghost:disabled{color:#9aa3af;border-color:var(--line);cursor:not-allowed;background:#fff}

@media(max-width:560px){.steps{flex-direction:column}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">法</div>
      <div>
        <h1>法律 RAG 问答</h1>
        <p class="sub">检索本地法条 → 大模型依据作答 → 合规校验</p>
      </div>
    </div>
    <span class="pill"><span class="dot"></span><span id="stat">加载中…</span></span>
  </header>

  <div class="card">
    <textarea id="q" placeholder="输入你的法律问题，例如：离婚冷静期是多少天？"></textarea>
    <div class="chips" id="ex"></div>
    <div class="row">
      <button class="primary" id="btn">提问</button>
      <label class="switch"><input type="checkbox" id="stream" checked> 流式输出</label>
      <label class="switch"><input type="checkbox" id="agent" checked> 多 Agent</label>
      <span class="spacer"></span>
      <label class="switch">检索
        <select id="k"><option value="3" selected>3</option><option value="5">5</option><option value="8">8</option></select>
      </label>
    </div>
  </div>

  <div class="card">
    <div class="steps" id="steps">
      <div class="step" data-i="0"><span class="n">1</span>意图分析</div>
      <div class="step" data-i="1"><span class="n">2</span>法条检索</div>
      <div class="step" data-i="2"><span class="n">3</span>依据作答</div>
      <div class="step" data-i="3"><span class="n">4</span>合规校验</div>
    </div>
    <div class="sec">
      <h2>回答</h2>
      <div style="display:flex;align-items:center;gap:4px">
        <span class="meta" id="badge"></span>
        <button class="mini" id="copy">复制</button>
      </div>
    </div>
    <div class="answer empty" id="answer">等待提问…</div>
  </div>

  <div class="card">
    <div class="sec">
      <h2>检索到的法条依据</h2>
      <span class="meta" id="srccount"></span>
    </div>
    <div id="sources"><div class="empty">暂无</div></div>
  </div>


  <div class="card">
    <div class="sec">
      <h2>知识库管理</h2>
      <span class="meta" id="kbstat">—</span>
    </div>
    <div class="drop" id="drop">把法条 .md 文件拖到这里，或点击选择文件</div>
    <input type="file" id="file" accept=".md,.txt" multiple hidden>
    <div class="kbrow">
      <button class="ghost" id="rebuild">重建索引</button>
      <span class="meta" id="kbmsg"></span>
    </div>
    <div class="flist" id="flist"></div>
  </div>

  <footer>答案仅依据本地知识库中的法条生成，不构成法律意见</footer>
</div>

<script>
fetch('/api/stats').then(r => r.json()).then(d => {
  document.getElementById('stat').textContent = '知识库 ' + d.total_chunks + ' 条 / ' + d.files.length + ' 部法律';
}).catch(() => { document.getElementById('stat').textContent = '统计接口异常'; });

const esc = s => String(s == null ? '' : s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const ansEl = document.getElementById('answer');
const srcEl = document.getElementById('sources');
const badge = document.getElementById('badge');
const srccount = document.getElementById('srccount');
const qEl = document.getElementById('q');
let rawAnswer = '';

const EXAMPLES = ['离婚冷静期是多少天？', '定金不得超过主合同标的额的多少？', '十三岁的小孩杀人要负刑事责任吗？', '精神病人发病时伤人要不要坐牢？'];
document.getElementById('ex').innerHTML = EXAMPLES.map(t => '<button class="chip">' + esc(t) + '</button>').join('');
document.getElementById('ex').querySelectorAll('.chip').forEach(b => { b.onclick = () => { qEl.value = b.textContent; qEl.focus(); }; });

const stepEls = document.querySelectorAll('.step');
function setStep(i) {
  stepEls.forEach(el => {
    const n = parseInt(el.dataset.i, 10);
    el.classList.toggle('on', n === i);
    el.classList.toggle('done', n < i);
  });
}

document.getElementById('copy').onclick = () => {
  if (!rawAnswer) return;
  navigator.clipboard.writeText(rawAnswer).then(() => {
    const b = document.getElementById('copy'); b.textContent = '已复制';
    setTimeout(() => { b.textContent = '复制'; }, 1500);
  });
};

function renderAnswer(done) {
  let h = esc(rawAnswer).replace(/第[一二三四五六七八九十百千零〇\\d]+条/g,
    m => '<span class="cite" data-a="' + m + '">' + m + '</span>');
  ansEl.className = 'answer';
  ansEl.innerHTML = h + (done ? '' : '<span class="caret"></span>');
  ansEl.querySelectorAll('.cite').forEach(el => {
    el.onclick = () => {
      const cards = document.querySelectorAll('.src');
      cards.forEach(c => c.classList.remove('hl'));
      for (const c of cards) {
        if ((c.dataset.article || '').includes(el.dataset.a)) {
          c.classList.add('hl');
          c.scrollIntoView({ behavior: 'smooth', block: 'center' });
          break;
        }
      }
    };
  });
}

function renderSources(sources, keywords) {
  if (!sources || !sources.length) {
    srcEl.innerHTML = '<div class="empty">未检索到相关法条</div>';
    srccount.textContent = '';
    return;
  }
  srccount.textContent = sources.length + ' 条';
  srcEl.innerHTML = sources.map((s, i) => {
    let text = esc(s.snippet);
    (keywords || []).forEach(w => {
      if (w && w.length > 1) text = text.split(esc(w)).join('<mark>' + esc(w) + '</mark>');
    });
    const pct = Math.max(4, Math.round(Math.min(1, s.score) * 100));
    return '<div class="src" data-article="' + esc(s.article || '') + '">' +
      '<div class="top"><span class="tag">' + (i + 1) + '</span>' +
      '<span class="tag law">' + esc(s.law_name) + '</span>' +
      (s.article ? '<span class="tag art">' + esc(s.article) + '</span>' : '') +
      '<span class="bar"><i style="width:' + pct + '%"></i></span>' +
      '<span class="meta">' + s.score.toFixed(2) + '</span></div>' +
      '<div class="text">' + text + '</div></div>';
  }).join('');
}

function handle(ev) {
  if (ev.type === 'stage') { setStep(ev.step); badge.textContent = ev.text; }
  else if (ev.type === 'sources') renderSources(ev.sources, ev.keywords);
  else if (ev.type === 'token') { rawAnswer += ev.text; renderAnswer(false); }
  else if (ev.type === 'reset') { rawAnswer = ''; renderAnswer(false); }
  else if (ev.type === 'score') {
    setStep(4);
    badge.textContent = '校验 ' + ev.score + ' · 有依据 ' + (ev.grounded ? '是' : '否') + ' · 重写 ' + ev.retry + ' 次';
  }
  else if (ev.type === 'error') { ansEl.className = 'err'; ansEl.textContent = ev.text; }
  else if (ev.type === 'done') renderAnswer(true);
}

const btn = document.getElementById('btn');
btn.onclick = async () => {
  const q = qEl.value.trim();
  const k = parseInt(document.getElementById('k').value, 10);
  const useStream = document.getElementById('stream').checked;
  const useAgent = document.getElementById('agent').checked;
  if (!q) return;

  btn.disabled = true; btn.textContent = '生成中…';
  rawAnswer = '';
  setStep(0); badge.textContent = '准备中…';
  ansEl.className = 'answer empty';
  ansEl.innerHTML = '<span class="caret"></span>';
  srcEl.innerHTML = '<div class="empty">等待检索…</div>';
  srccount.textContent = '';

  try {
    if (useStream) {
      const res = await fetch('/api/ask/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, top_k: k })
      });
      const reader = res.body.getReader();
      const dec = new TextDecoder('utf-8');
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split('\\n');
        buf = parts.pop();
        for (const line of parts) {
          if (!line.startsWith('data: ')) continue;
          try { handle(JSON.parse(line.slice(6))); } catch (e) {}
        }
      }
    } else {
      const res = await fetch(useAgent ? '/api/ask/agent' : '/api/ask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, top_k: k })
      });
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail || '请求失败');
      rawAnswer = d.answer;
      renderSources(d.sources, []);
      setStep(4);
      renderAnswer(true);
      badge.textContent = d.mode === 'multi-agent'
        ? '校验 ' + d.score + ' · 有依据 ' + (d.grounded ? '是' : '否') + ' · 重写 ' + d.retry + ' 次'
        : '单 Agent 模式';
    }
  } catch (e) {
    ansEl.className = 'err'; ansEl.textContent = '出错了：' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = '提问';
  }
};


const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const flist = document.getElementById('flist');
const kbmsg = document.getElementById('kbmsg');
const kbstat = document.getElementById('kbstat');
const rebuildBtn = document.getElementById('rebuild');

drop.onclick = () => fileInput.click();
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files.length) uploadAll(e.dataTransfer.files); });
fileInput.onchange = () => { if (fileInput.files.length) uploadAll(fileInput.files); fileInput.value = ''; };

async function uploadAll(files) {
  for (const f of files) {
    kbmsg.textContent = '上传中：' + f.name;
    const fd = new FormData(); fd.append('file', f);
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || '上传失败');
      kbmsg.textContent = '已上传 ' + d.name;
    } catch (e) {
      kbmsg.textContent = '上传失败：' + e.message;
      break;
    }
  }
  await loadFiles();
  kbmsg.textContent = '上传完成，正在重建索引…';
  await doRebuild();
}

async function loadFiles() {
  try {
    const d = await (await fetch('/api/files')).json();
    flist.innerHTML = d.files.map(f =>
      '<div class="fitem"><span class="nm">' + esc(f.name) + '</span>' +
      '<span class="sz">' + Math.max(1, Math.round(f.size / 1024)) + ' KB</span>' +
      '<button class="del" data-n="' + esc(f.name) + '" title="删除">×</button></div>'
    ).join('') || '<div class="empty">暂无文件</div>';
    flist.querySelectorAll('.del').forEach(b => {
      b.onclick = async () => {
        if (!confirm('确定删除 ' + b.dataset.n + ' ？')) return;
        await fetch('/api/files/' + encodeURIComponent(b.dataset.n), { method: 'DELETE' });
        await loadFiles();
        kbmsg.textContent = '已删除，记得重建索引';
      };
    });
    kbstat.textContent = d.count + ' 个文件';
  } catch (e) { kbstat.textContent = '读取失败'; }
}

async function doRebuild() {
  rebuildBtn.disabled = true;
  const old = rebuildBtn.textContent;
  rebuildBtn.textContent = '重建中…';
  kbmsg.textContent = '正在重新切片并向量化，请稍候…';
  try {
    const d = await (await fetch('/api/rebuild', { method: 'POST' })).json();
    kbmsg.textContent = '重建完成，共 ' + d.total_chunks + ' 条切片';
    const st = await (await fetch('/api/stats')).json();
    document.getElementById('stat').textContent = '知识库 ' + st.total_chunks + ' 条 / ' + st.files.length + ' 部法律';
  } catch (e) {
    kbmsg.textContent = '重建失败：' + e.message;
  } finally {
    rebuildBtn.disabled = false;
    rebuildBtn.textContent = old;
  }
}

rebuildBtn.onclick = doRebuild;
loadFiles();

qEl.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') btn.click();
});
</script>
</body>
</html>
"""


# ---------------- 接口 ----------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/api/stats")
def stats():
    files = sorted(p.stem for p in Path(DATA_DIR).rglob("*.md"))
    return {
        "total_chunks": _state.get("size", 0),
        "files": files,
        "vector_store": str(STORE_DIR),
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    retriever = _state["retriever"]
    retriever.k = req.top_k

    docs = retriever.invoke(question)
    answer = _state["chain"].invoke(question)

    return {
        "mode": "single",
        "answer": answer,
        "sources": [
            {
                "law_name": d.metadata.get("law_name", ""),
                "score": round(d.metadata.get("score", 0.0), 4),
                "snippet": d.page_content,
            }
            for d in docs
        ],
    }


@app.post("/api/ask/agent")
def ask_agent(req: AskRequest):
    """多 Agent 模式：意图分析 -> 检索 -> 作答 -> 合规校验（不通过自动重写一次）。"""
    from multi_agent_runner import run_pipeline

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    r = run_pipeline(question, k=req.top_k, verbose=False)
    return {
        "mode": "multi-agent",
        "answer": r["answer"],
        "score": r["score"],
        "grounded": r["grounded"],
        "retry": r["retry"],
        "issues": r["issues"],
        "sources": r["sources"],
    }


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """
    SSE 流式问答：一步步把 Agent 的中间结果推给前端。
    事件顺序：stage(意图分析) → sources(检索到的法条) → token(答案逐段) → score(校验) → done
    """
    from multi_agent_runner import run_pipeline_stream

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    def gen():
        try:
            for ev in run_pipeline_stream(question, k=req.top_k):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:                       # 生成过程中出错也要通知前端
            yield f"data: {json.dumps({'type': 'error', 'text': f'{type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/files")
def list_files():
    """列出 data/ 下已有的法条文件。"""
    items = []
    for p in sorted(Path(DATA_DIR).rglob("*.md")):
        items.append({"name": p.name, "size": p.stat().st_size})
    return {"files": items, "count": len(items)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """
    上传一个法条文件到 data/。

    安全考虑：
      - 只认 .md / .txt，且用 Path().name 剥掉客户端可能传来的路径，防目录穿越
      - 限制大小，避免大文件把内存打爆
      - 强制 UTF-8，中文法条常见 GBK 编码会在这里被拦下并给出明确提示
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    name = Path(file.filename).name
    if not name.lower().endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="只支持 .md 或 .txt 文件")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"文件超过 {MAX_UPLOAD_MB}MB")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="文件不是 UTF-8 编码。请用记事本/VS Code 另存为 UTF-8 后重试",
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    target = Path(DATA_DIR) / name
    target.write_text(text, encoding="utf-8")
    return {"status": "ok", "name": name, "size": len(content)}


@app.delete("/api/files/{name}")
def delete_file(name: str):
    """删除 data/ 下的某个法条文件（不会自动重建索引，需再点一次重建）。"""
    data_root = Path(DATA_DIR).resolve()
    target = (data_root / Path(name).name).resolve()
    if not str(target).startswith(str(data_root)):   # 防目录穿越
        raise HTTPException(status_code=400, detail="非法文件名")
    if not target.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        target.unlink()
    except OSError as e:
        # 文件被占用、权限不足或系统回收站不可用时，给前端明确提示而不是 500
        raise HTTPException(status_code=500, detail=f"删除失败：{e}")
    return {"status": "ok", "name": target.name}


@app.post("/api/rebuild")
def rebuild():
    """重新读取 data/ 并重建向量库。改动法条数据后调用。"""
    try:
        chunks = load_and_split()
        SimpleVectorStore.from_documents(
            documents=chunks,
            embedding=build_embeddings(),
            persist_directory=STORE_DIR,
            collection_name="chinese_law",
        )
        n = _refresh()
        return JSONResponse({"status": "ok", "total_chunks": n})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建失败：{type(e).__name__}: {e}")
