# 03 環境與操作

← 回到 [主檔](00-main.md)

## 1. 硬體與軟體

| 項目 | 規格 / 版本 | 備註 |
|---|---|---|
| CPU | Intel i7-1165G7（4 核 8 緒） | Ollama 以 CPU 推論 |
| RAM | 16 GB | 3B～8B（4-bit）可載入；14B 以上不建議 |
| GPU | Intel Iris Xe（內顯） | 視同無 GPU |
| 磁碟 | C 槽可用約 52 GB | 模型合計約 3 GB（含 7B 約 8 GB） |
| Python | 3.13（專案 `.venv`） | |
| Ollama | 0.40.10 | |
| Git | 2.56.0 | 安裝需系統管理員權限（UAC） |

使用提醒：執行時**接上電源**，少開瀏覽器分頁（RAM 吃緊會變慢）。

## 2. 模型

| 用途 | 模型 | 大小 | 狀態 |
|---|---|---|---|
| 對話（預設） | `qwen2.5:3b` | 1.9 GB | ✅ 已下載 |
| 對話（品質比較） | `qwen2.5:7b` | 4.7 GB | 需要時再 `ollama pull` |
| Embedding | `bge-m3` | 1.2 GB | ✅ 已下載 |

## 3. 環境重建步驟

```powershell
# 1. 安裝工具（Git 會跳出 UAC，需按「是」）
winget install Ollama.Ollama
winget install Git.Git

# 2. Git 作者資訊（新開終端機視窗後執行一次）
git config --global user.name "你的名字"
git config --global user.email "you@example.com"

# 3. 下載模型
ollama pull qwen2.5:3b
ollama pull bge-m3

# 4. Python 虛擬環境
cd C:\projects\rag-chatbot
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## 4. 常用指令

```powershell
.\.venv\Scripts\Activate.ps1

# 命令列工具（M2 起；完整說明見 02 §4.4）
python -X utf8 -m rag_chatbot profiles                      # 列出情境
python -X utf8 -m rag_chatbot ingest                        # 匯入 data\collections\ 下全部文件集
python -X utf8 -m rag_chatbot list                          # 列出已匯入文件
python -X utf8 -m rag_chatbot search "問題" --profile intro  # 檢視檢索結果與分數
python -X utf8 -m rag_chatbot delete me 履歷.docx            # 刪除文件
python -X utf8 -m rag_chatbot ask "問題" --profile intro     # 提問一題（M3 起）
python -X utf8 -m rag_chatbot chat --sources                # 互動式連續提問（M3 起）

# 測試
pytest                                                      # 單元測試
pytest --cov=rag_chatbot                                    # 含覆蓋率

streamlit run app.py                                        # 管理介面（M5 起）
python -X utf8 -m rag_chatbot.eval eval\questions.csv       # 批次評估（M6 起）
```

> Windows 主控台預設 Big5（cp950），命令列輸出中文需加 `-X utf8`。

**放置文件**：把文件放進 `data\collections\<文件集>\`（例如 `data\collections\me\`），再執行 `ingest`。文件集內容視為可公開（N-06）。

## 5. 效能實測（2026-10-24）

模擬 RAG：送入約 1,400 tokens 參考資料 + 問題，使用 `qwen2.5:3b`。

| 項目 | 結果 |
|---|---|
| 模型載入（首次） | 3.7 秒 |
| 讀入參考資料 | 1,438 tokens / 16.6 秒（約 87 tok/s） |
| 產生回答 | 約 10 tok/s |
| **首次回答總時間** | **19.4 秒**（符合 N-02） |
| 相同內容再問（Ollama 快取） | 2.5 秒 |
| bge-m3 embedding（短句） | 約 0.43 秒/段 |
| bge-m3 embedding（約 600 字/段，2026-10-25） | 約 0.9 秒/段；每批 1 筆或 16 筆速度相同；1,000 段約 15 分鐘 |
| 向量搜尋（1 萬段 × 1024 維，2026-10-25） | 約 100 ms（含從 SQLite 讀取）；資料庫約 62 MB |
| 真實文件 RAG 回答（M3 規劃前，2026-10-25） | 送入約 2,000 字參考資料：讀入 1,962 tokens／36 秒（約 54 tok/s）+ 產生 199 tokens／25 秒，**共 70 秒（超過 N-02）**；拒答時約 26 秒 → M3 以參考資料上限、回答長度上限、keep_alive 改善（D-024） |
| M3 完成後回答時間（2026-10-25） | 測試資料 7.6～15.9 秒；真實文件 14～24 秒；問候 0 秒（符合 N-02） |

觀察：
- 回答內容正確，但模型未照指示標註 `[n]`（→ D-010）。
- CPU 上分批送出不會加速；`OllamaEmbedder` 仍每批 16 筆，用於回報匯入進度。

## 6. 提問測試（命令列，M3 起；網頁介面於 M5 提供）

**事前準備**：Ollama 在執行中（工作列有 Ollama 圖示；或執行 `ollama list` 有回應）、筆電接上電源。

```powershell
cd C:\projects\rag-chatbot
.\.venv\Scripts\Activate.ps1

# 方式一：互動式對話（建議）——同一個對話中連續追問
python -X utf8 -m rag_chatbot chat --sources
#   您：她管理過多大的團隊？
#   [回答｜18.2 秒｜Prompt v1｜語意最高分 0.641]
#   ……回答內容……
#   回答依據：
#     [1] 0.641  me/常見問題_進階 > 高階主管能力與組織領導
#   您：exit            ← 輸入 exit 或直接按 Enter 離開

# 方式二：一次問一題
python -X utf8 -m rag_chatbot ask "智慧訂單可以接哪些 ERP？"
python -X utf8 -m rag_chatbot ask "那導入要多久？" --conversation 3   # 接續對話 #3 追問

# 指定情境（預設 intro）
python -X utf8 -m rag_chatbot chat --profile me
```

**看懂回答開頭的標示**

| 標示 | 意思 |
|---|---|
| `回答` | 模型依文件回答 |
| `拒答` | 文件中沒有答案（秒回 = 門檻擋下；十幾秒以上 = 模型判斷） |
| `問候` | 問候、道謝、道別、套話的固定回覆（不呼叫模型） |
| `語意最高分` | 問題與最相關段落的分數，用於調整門檻（§7） |

**測試時建議記錄**：答錯（內容與文件不符）、不該拒答卻拒答、該拒答卻回答、回答超過 60 秒的題目，連同畫面上的標示一起記下，供調整 Prompt、門檻或補充 FAQ。所有問答也都保存在資料庫的對話紀錄中。

**修改文件後**：執行 `python -X utf8 -m rag_chatbot ingest` 重新匯入，不需重新啟動任何程式。

## 7. 調整拒答門檻

`min_score`（預設 0.40）：問題與最相關段落的語意分數（0～1）低於此值時，直接回拒答文字、不呼叫 LLM。它只負責擋掉明顯無關的問題；其餘能否回答由 LLM 判斷（D-022）。

**何時要調整**

| 現象 | 代表 | 調整 |
|---|---|---|
| 文件有答案，卻在幾秒內就回拒答文字（未呼叫 LLM） | 門檻太高，誤擋相關問題 | 調低 |
| 無關問題被送進 LLM，回答答非所問，或等很久才拒答 | 門檻太低 | 調高 |

**檢查時機**：每個里程碑的功能測試時；開放 LINE 試用後，定期檢視對話紀錄中被拒答的問題（`messages.kind`、`messages.top_score`）。

**調整步驟**

1. 收集「應該回答」與「應該拒答」的問題各數題。
2. 以 `python -X utf8 -m rag_chatbot search "問題"` 查看每題第 1 名的分數（`ask`／`chat` 的回答開頭也會顯示；M5 起管理介面也會顯示）。
3. 把門檻設在兩組分數之間：修改 `config.toml` 的 `[retrieval] min_score`；只想調整某個情境時，在 `profiles/<情境>.toml` 加 `min_score = 0.42`。不需改程式、不需重新匯入。
4. 驗證：重跑這些問題（M6 起可用批次評估整批重跑），確認沒有讓其他題目變差。
5. 在 [04 §4](04-testing.md#4-執行紀錄) 記錄調整前後的分數與理由。
