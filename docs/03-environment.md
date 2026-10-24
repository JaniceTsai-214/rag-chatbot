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
python -X utf8 -m rag_chatbot ask "問題" --profile intro     # 提問（M3 起）

# 測試
pytest                                                      # 單元測試
pytest --cov=rag_chatbot                                    # 含覆蓋率

streamlit run app.py                                        # 管理介面（M4 起）
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

觀察：
- 回答內容正確，但模型未照指示標註 `[n]`（→ D-010）。
- CPU 上分批送出不會加速；`OllamaEmbedder` 仍每批 16 筆，用於回報匯入進度。
