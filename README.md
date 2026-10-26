# rag-chatbot

**用 LINE 問 AI「這個人做過什麼」，AI 依履歷與作品文件回答。**
A local RAG chatbot that answers questions about a person and their products from their own documents, served via LINE.

以自有文件（PDF / Word / Markdown / FAQ）為知識來源的 RAG 問答系統。訪客在 LINE 官方帳號提問，系統從文件中找出依據、交給本機 LLM（Ollama）回答，資料不出筆電、沒有 API 費用。

<!-- TODO：放 LINE 對話截圖或 GIF，例如 docs/images/line-demo.gif -->
<!-- TODO：示範影片連結、LINE 官方帳號 QR code -->

| | |
|---|---|
| 語言 / 框架 | Python 3.11+、FastAPI、SQLite、numpy |
| 模型 | 對話 `qwen2.5:3b`、Embedding `bge-m3`（皆透過 Ollama 在本機 CPU 執行） |
| 測試 | 329 項單元測試，核心覆蓋率 99%；每個里程碑另有功能測試紀錄 |
| 文件 | 需求、設計、環境、測試文件，以及 30 多條決策紀錄（[docs/00-main.md](docs/00-main.md)） |

---

## 功能

- **文件匯入**：PDF（記錄頁碼）、DOCX（依標題分節、表格轉文字）、Markdown、FAQ（CSV / XLSX）。以檔案 hash 判斷是否需要重新匯入，匯入以單一交易完成，中途失敗不留下半套資料。
- **混合檢索**：語意（bge-m3 向量）＋ 關鍵字（BM25，中文 bigram），以 RRF 合併排名。
- **有依據的回答**：每段參考資料標示出處（檔名、標題、頁碼），Prompt 要求只依參考資料回答；文件沒有提到時說明並引導來信。
- **情境（profile）**：一份設定檔決定「查哪些文件、用什麼 Prompt、怎麼拒答」。換一組文件與設定即可支援其他問答，不需改程式。
- **LINE 串接**：一個指令啟動 webhook 與 Cloudflare 通道，並自動更新 LINE 的 Webhook 網址；簽章驗證、多輪對話、頻率限制、逾時改用 push 補送、手機易讀的純文字排版。
- **管理與追蹤**：命令列工具（匯入、檢索分數、提問、互動對話）、Prompt 版本管理、完整對話紀錄（含分數與耗時，用來校正門檻與 Prompt）。

## 架構

```
訪客手機 ─▶ LINE Platform ─▶ Cloudflare 通道 ─▶ FastAPI /callback ─┐
                                                                  │
管理者 ─▶ 命令列 (cli.py) ─────────────────────────────────────────┤
                                                                  ▼
                                 RagService.answer(question, profile)   ← 所有介面共用
                                                                  │
   問候/閒聊判斷 ─▶ 混合檢索 ─▶ 門檻 ─▶ 組 Prompt ─▶ LLM ─▶ 後處理 ─▶ 對話紀錄
                       │                            │
        Embedder (bge-m3) + VectorStore        LLM 介面（Ollama 實作）
        (SQLite + numpy) + BM25

匯入：文件集資料夾 ─▶ Loaders ─▶ Chunker ─▶ Embedder ─▶ VectorStore
```

`answer()` 與介面分離，LLM、Embedding、向量儲存都是抽象介面：LINE 與命令列共用同一個核心，單元測試用假物件、不需要 Ollama。詳見 [docs/02-design.md](docs/02-design.md)。

## 技術取捨（節錄）

完整理由見 [決策紀錄](docs/00-main.md#5-決策紀錄)。

| 決策 | 理由 |
|---|---|
| 本機 Ollama，不用雲端 API（D-001、D-011） | 資料不出筆電、零費用；實測首次回答約 19 秒，可接受 |
| SQLite + numpy，不用向量資料庫（D-005） | 個人文件量小：1 萬段全部比對約 100 ms。向量儲存已抽象化，規模變大時可換成 pgvector |
| 混合檢索（D-021） | 真實文件實測：只用語意檢索時，「做過哪些產品」這類列舉題找不齊 |
| 門檻只擋明顯無關的問題（D-022） | 真實文件中相關與無關問題的分數重疊，單靠門檻會誤擋，改由 LLM 依 Prompt 判斷 |
| 程式刪除多餘的拒答句（D-028） | 3B 模型常先抄一句拒答句、再給出正確答案；整段當拒答會丟掉正確內容 |
| 只轉換含簡體字的句子（D-027） | 3B 會輸出簡體字；整段轉換會把「天行智能」改成「天行智慧」、「干擾」改成「幹擾」 |
| 問候、套話以規則判斷（D-023） | 0 秒回應、不浪費模型時間，也避免對「你好」回拒答 |
| 不用 line-bot-sdk（D-030） | 需要的 API 只有 5 個，自行實作簽章驗證與呼叫，較精簡、好測試 |

## 快速開始

需要：Python 3.11 以上、[Ollama](https://ollama.com)。以下以 Windows 為例。

```powershell
# 1. 下載模型
ollama pull qwen2.5:3b
ollama pull bge-m3

# 2. 建立虛擬環境並安裝
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 3. 執行單元測試（不需要 Ollama）
.venv\Scripts\python.exe -m pytest --cov=rag_chatbot

# 4. 放入文件並匯入：文件放在 data\collections\<文件集>\，例如 data\collections\me\
.venv\Scripts\python.exe -X utf8 -m rag_chatbot ingest

# 5. 提問
.venv\Scripts\python.exe -X utf8 -m rag_chatbot chat --sources
```

`-X utf8` 讓 Windows 主控台正確顯示中文。沒有自己的文件時，可以用 [tests/functional/collections/](tests/functional/collections/) 的虛構測試資料（人物「王小明」、產品「小算盤記帳 App」）。

其他指令（`profiles`、`list`、`search`、`delete`、`ask`）與 LINE 機器人的設定步驟，見 [docs/03-environment.md](docs/03-environment.md)。

## 開發方式

- **文件先行**：需求與設計先寫進文件並確認，才開始寫程式。方向改變時新增一條決策紀錄。
- **完成的定義**：程式完成且單元測試全部通過；每個里程碑結束時執行功能測試並記錄結果（[docs/04-testing.md](docs/04-testing.md)）。
- **以真實文件校正**：門檻、Prompt、後處理規則都依實測結果調整，校正紀錄保留在測試文件中。

| 里程碑 | 內容 | 狀態 |
|---|---|---|
| M0～M1 | 需求、設計、環境 | ✅ |
| M2 | 文件匯入（解析、切段、向量化、儲存、命令列） | ✅ |
| M3 | 檢索與回答核心 `answer()` | ✅ |
| M4 | LINE 串接 | ✅ |
| M5 | Streamlit 管理介面 | 規劃中 |
| M6 | 批次評估工具（含 3B／7B 品質比較） | 規劃中 |

## 專案結構

```
src/rag_chatbot/
├─ loaders/          文件解析：pdf、docx、markdown、faq
├─ chunker.py        切段（字元數、句界斷開、重疊）
├─ embedder.py       Embedding 介面 + Ollama 實作
├─ vector_store.py   SQLite 段落與向量、相似度搜尋
├─ keyword_index.py  BM25 關鍵字檢索
├─ retriever.py      混合檢索、門檻、參考資料長度上限
├─ rag.py            answer() 核心與後處理
├─ smalltalk.py      問候、閒聊、套話判斷
├─ prompts.py        Prompt 版本管理
├─ history.py        對話紀錄
├─ line/             LINE webhook、簽章、API、訪客與佇列、啟動流程
└─ cli.py            命令列工具
profiles/            情境設定
docs/                需求、設計、環境、測試、LINE 串接文件
tests/               單元測試與功能測試資料
```

## 作者

蔡佳惠（Janice Tsai）・tsaichiahui@gmail.com
