# 02 設計

← 回到 [主檔](00-main.md)

## 1. 架構總覽

```
            ┌──────────────── 第一階段 ────────────────┐   ┌──── 第二階段 ────┐
使用者 ──▶  Streamlit 介面 (app.py)                        LINE ─▶ HTTPS 通道 ─▶ FastAPI webhook
            │                                              │
            └────────────────┬─────────────────────────────┘
                             ▼
                    核心：answer(question, ...)  ← 兩階段共用
                             │
        ┌────────────┬───────┴───────┬──────────────┐
        ▼            ▼               ▼              ▼
   Retriever     PromptStore      LLM (Ollama)   HistoryStore
        │
   Embedder (Ollama bge-m3) + VectorStore (SQLite + numpy)
        ▲
   Ingest：Loaders (PDF/DOCX/MD/FAQ) → Chunker → Embedder → VectorStore
```

## 2. 目錄結構

```
rag-chatbot/
├─ README.md
├─ docs/                    # 本文件群
├─ config.toml              # 模型、參數設定
├─ pyproject.toml           # 依賴與 pytest 設定
├─ data/
│  ├─ inbox/                # 放要匯入的文件
│  └─ app.db                # SQLite：段落、向量、Prompt、對話紀錄（不進 Git）
├─ eval/
│  └─ questions.csv         # 批次評估題集
├─ src/rag_chatbot/
│  ├─ config.py             # 讀取 config.toml
│  ├─ models.py             # 資料類別：Document, Chunk, Source, Answer
│  ├─ loaders/              # pdf.py, docx.py, markdown.py, faq.py
│  ├─ chunker.py            # 切段
│  ├─ embedder.py           # Embedder 介面 + OllamaEmbedder
│  ├─ llm.py                # LLM 介面 + OllamaLLM
│  ├─ db.py                 # SQLite 連線與資料表建立
│  ├─ vector_store.py       # 段落與向量的存取、相似度搜尋
│  ├─ ingest.py             # 匯入流程（含檔案 hash 判斷）
│  ├─ retriever.py          # 檢索 + 門檻過濾
│  ├─ prompts.py            # Prompt 版本管理
│  ├─ history.py            # 對話紀錄
│  ├─ rag.py                # answer() 核心
│  └─ eval.py               # 批次評估
├─ app.py                   # Streamlit 入口
└─ tests/
   ├─ fixtures/             # 單元測試用小型 PDF/DOCX/MD/CSV
   ├─ functional/           # 功能測試用文件（見 04 §2.1）
   ├─ fakes.py              # FakeEmbedder, FakeLLM
   └─ test_*.py
```

## 3. 主要流程

### 3.1 匯入
1. 讀檔 → 計算 SHA-256。若同路徑且 hash 相同 → 跳過；hash 不同 → 刪除舊段落後重新匯入。
2. Loader 依副檔名解析，輸出含「位置資訊」的文字區塊：
   - PDF：逐頁（記錄頁碼）
   - DOCX：依標題樣式分節（記錄標題）
   - MD：依 `#` 標題分節（記錄標題路徑，如「洗衣機 > 保養」）
   - FAQ：每列一筆，不再切段，內容為「問：… 答：…」
3. Chunker：每段約 600 字、重疊 100 字，優先在段落/句號處斷開。
4. Embedder 批次轉向量 → 存入 SQLite。

### 3.2 回答 `answer()`
1. （多輪時）將最近 2 輪對話 + 新問題組成檢索查詢。
2. 問題轉向量 → 與所有段落算餘弦相似度 → 取 top_k（預設 5）。
3. 過濾低於門檻（預設 0.45，實測後調整）的段落。**全部低於門檻 → 直接回固定拒答文字，不呼叫 LLM。**
4. 組 Prompt：System Prompt（目前版本）+ 編號段落 `[1]…[5]` + 問題。
5. 呼叫 Ollama，回傳 `Answer`（文字、依據段落、模型、Prompt 版本、耗時），寫入對話紀錄。
6. 「回答依據」= 第 3 步送給模型的段落，不依賴模型自行標註 `[n]`（D-010）。

### 3.3 介面（Streamlit）
- 左側欄：對話清單、新對話、文件管理（上傳 / 列表 / 刪除 / 重新匯入）、Prompt 編輯與版本切換、參數（top_k、門檻、模型）。
- 主畫面：聊天訊息；每則回答下方可展開「回答依據」（檔名、頁碼/標題、分數、原文）；「用目前 Prompt 重跑」按鈕，結果並排顯示。

## 4. 設計取捨

| 項目 | 選擇 | 替代方案 | 理由 |
|---|---|---|---|
| Embedding | Ollama `bge-m3` | sentence-transformers 本機跑 | 不必安裝 PyTorch（~2GB），與 LLM 同一個執行環境 |
| 向量儲存 | SQLite + numpy | Chroma、FAISS | 數萬段內暴力搜尋 < 100ms；單一檔案好備份；無額外服務、相容 Python 3.13 |
| 介面 | Streamlit | Gradio、自寫前端 | 聊天元件 + 側欄最省事 |
| 抽象化 | `Embedder`、`LLM` 為 Protocol | 直接呼叫 Ollama | 單元測試用 Fake，不需 Ollama；日後換模型/換雲端只改一處 |
| 拒答 | 門檻過濾 + Prompt 約束雙重把關 | 只靠 Prompt | 小模型容易不聽話，先在程式層擋掉 |

## 5. 資料結構（SQLite）

```sql
documents(id PK, path, filename, file_type, sha256, imported_at)
chunks(id PK, document_id FK, chunk_index, text, page, heading, embedding BLOB)
prompts(id PK, version INT, content, note, created_at, is_active)
conversations(id PK, title, created_at)
messages(id PK, conversation_id FK, role, content,
         prompt_version, model, sources_json, latency_ms, created_at,
         rerun_of FK NULL)          -- 重跑時指向原訊息
```

## 6. 核心介面

```python
@dataclass
class Source:
    chunk_id: int
    filename: str
    page: int | None
    heading: str | None
    text: str
    score: float

@dataclass
class Answer:
    text: str
    sources: list[Source]
    refused: bool          # 是否因檢索不到而拒答
    model: str
    prompt_version: int
    latency_ms: int

def answer(question: str, conversation_id: int | None = None,
           prompt_version: int | None = None) -> Answer: ...
```

## 7. 預設 System Prompt（v1，可在介面修改）

```
你是個人知識助理。請「只根據」下方提供的參考資料回答問題。
規則：
1. 使用繁體中文，簡潔清楚，優先條列，約 300 字以內。
2. 每個重點後面標註引用的資料編號，例如 [1]、[2]。
3. 如果參考資料中沒有答案，請回答：「目前的資料中找不到相關資訊。」不要猜測或補充資料外的內容。
```

## 8. 第二階段預留（概要，屆時另開文件）

- `line_webhook.py`：FastAPI `/callback`，驗證 `X-Line-Signature`，收到文字訊息 → `answer()` → reply。
- 每位 LINE 使用者對應一個 `conversation`，沿用同一套對話紀錄。
- CPU 推論較慢，需先回應 LINE（或顯示載入動畫）再處理，避免 reply token 逾時；屆時實測決定做法。
- 通道：Cloudflare Tunnel 或 ngrok，只開放 webhook 埠；Streamlit 管理介面不對外。
