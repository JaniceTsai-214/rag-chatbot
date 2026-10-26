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
winget install Cloudflare.cloudflared          # LINE 機器人的 HTTPS 通道（M4）

# 2. Git 作者資訊（新開終端機視窗後執行一次）
git config --global user.name "你的名字"
git config --global user.email "you@example.com"

# 3. 下載模型
ollama pull qwen2.5:3b
ollama pull bge-m3

# 4. Python 虛擬環境與套件
cd C:\MyProjects\rag-chatbot
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 5. 匯入文件（文件放在 data\collections\，見 §4）
.venv\Scripts\python.exe -X utf8 -m rag_chatbot ingest
```

另需建立 `.env`（LINE 金鑰，見 §7）。`data\`、`.env` 不在 Git 中，換電腦時需自行備份與複製；備份放在 `E:\MyProjects\rag-chatbot`（D-031）。

## 4. 常用指令

所有指令都在專案資料夾執行（`cd C:\MyProjects\rag-chatbot`），開頭一律是 `.venv\Scripts\python.exe`：命令提示字元（cmd）與 PowerShell 皆適用，不需先啟動虛擬環境。

```powershell
# 命令列工具（完整說明見 02 §4.4）
.venv\Scripts\python.exe -X utf8 -m rag_chatbot profiles                      # 列出情境
.venv\Scripts\python.exe -X utf8 -m rag_chatbot ingest                        # 匯入 data\collections\ 下全部文件集
.venv\Scripts\python.exe -X utf8 -m rag_chatbot list                          # 列出已匯入文件
.venv\Scripts\python.exe -X utf8 -m rag_chatbot search "問題" --profile intro  # 檢視檢索結果與分數
.venv\Scripts\python.exe -X utf8 -m rag_chatbot delete me 履歷.docx            # 刪除文件
.venv\Scripts\python.exe -X utf8 -m rag_chatbot ask "問題" --profile intro     # 提問一題
.venv\Scripts\python.exe -X utf8 -m rag_chatbot chat --sources                # 互動式連續提問
.venv\Scripts\python.exe -X utf8 -m rag_chatbot line                          # 啟動 LINE 機器人（§7）

# 測試
.venv\Scripts\python.exe -m pytest                                            # 單元測試
.venv\Scripts\python.exe -m pytest --cov=rag_chatbot                          # 含覆蓋率
```

> - 出現「No module named rag_chatbot」：用到了系統的 Python，請確認指令開頭是 `.venv\Scripts\python.exe`。
> - `-X utf8`：Windows 主控台預設 Big5（cp950），加上它中文才能正確顯示。

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
cd C:\MyProjects\rag-chatbot

# 方式一：互動式對話（建議）——同一個對話中連續追問
.venv\Scripts\python.exe -X utf8 -m rag_chatbot chat --sources
#   您：她管理過多大的團隊？
#   [回答｜18.2 秒｜Prompt v1｜語意最高分 0.641]
#   ……回答內容……
#   回答依據：
#     [1] 0.641  me/常見問題_進階 > 高階主管能力與組織領導
#   您：exit            ← 輸入 exit 或直接按 Enter 離開

# 方式二：一次問一題
.venv\Scripts\python.exe -X utf8 -m rag_chatbot ask "智慧訂單可以接哪些 ERP？"
.venv\Scripts\python.exe -X utf8 -m rag_chatbot ask "那導入要多久？" --conversation 3   # 接續對話 #3 追問

# 指定情境（預設 intro）
.venv\Scripts\python.exe -X utf8 -m rag_chatbot chat --profile me
```

**看懂回答開頭的標示**

| 標示 | 意思 |
|---|---|
| `回答` | 模型依文件回答 |
| `拒答` | 文件中沒有答案（秒回 = 門檻擋下；十幾秒以上 = 模型判斷） |
| `問候` | 問候、道謝、道別、套話的固定回覆（不呼叫模型） |
| `語意最高分` | 問題與最相關段落的分數，用於調整門檻（§7） |

**測試時建議記錄**：答錯（內容與文件不符）、不該拒答卻拒答、該拒答卻回答、回答超過 60 秒的題目，連同畫面上的標示一起記下，供調整 Prompt、門檻或補充 FAQ。所有問答也都保存在資料庫的對話紀錄中。

**修改文件後**：執行 `.venv\Scripts\python.exe -X utf8 -m rag_chatbot ingest` 重新匯入。命令列與 LINE 機器人都不需重新啟動（下一題起自動使用新內容）。

**修改情境設定（`profiles\*.toml`、`config.toml`）後**：LINE 機器人需重新啟動（Ctrl+C 後再執行）。`system_prompt` 的修改不影響已存入資料庫的 Prompt 版本（見 02 §8）。

## 7. LINE 機器人（M4）

**第一次設定：建立 `.env`**（金鑰只放這裡，不要貼到對話或其他檔案；`.env` 不會進 Git）

1. 在 VS Code 左側檔案總管，於專案根目錄（`config.toml` 旁）新增檔案，檔名**就是 `.env`**（前面不加任何文字，其他檔名讀不到）。
2. 內容兩行（等號後面直接貼上，不加空白或引號）：
   ```
   LINE_CHANNEL_SECRET=貼上 Channel secret
   LINE_CHANNEL_ACCESS_TOKEN=貼上 Channel access token
   ```
   - Channel secret：LINE Developers Console →「Basic settings」
   - Channel access token：LINE Developers Console →「Messaging API」最下方
3. 存檔。

**第一次啟動後（只需一次）**：若畫面提示「Use webhook 尚未開啟」，到 LINE Developers Console →「Messaging API」→「Webhook settings」打開「Use webhook」（與官方帳號後台「回應設定」的 Webhook 開關是同一設定）。

### 7.1 每次啟動（含重新開機後）

| 步驟 | 操作 | 確認 |
|---|---|---|
| 1 | 筆電接上電源 | — |
| 2 | 確認 Ollama 在執行：登入 Windows 後會自動啟動（啟動資料夾中有 `Ollama.lnk`） | 工作列右下角有 Ollama 圖示；沒有時從「開始」選單開啟 Ollama |
| 3 | 開啟終端機（命令提示字元、PowerShell，或 VS Code 的終端機皆可） | — |
| 4 | 輸入下方兩行指令 | — |
| 5 | 等待約 1 分鐘 | 出現「LINE 機器人已上線」與「連線測試：成功」 |
| 6 | 用手機傳「你好」給官方帳號 | 立即收到問候文字；終端機出現一行紀錄 |

```powershell
cd C:\MyProjects\rag-chatbot
.venv\Scripts\python.exe -X utf8 -m rag_chatbot line
```

啟動過程依序顯示：

```
檢查 Ollama 並載入模型…
建立 Cloudflare 通道…
等待通道網址生效：https://xxxx.trycloudflare.com
LINE 機器人已上線（情境 intro）
Webhook 網址：https://xxxx.trycloudflare.com/callback
連線測試：成功
按 Ctrl+C 停止。
```

之後每則問答列一行，例如 `21:05:12 訪客…a1b2 answered 18.3 秒 reply｜她管理過多大的團隊？`（時間、訪客 ID 末 4 碼、類型、耗時、reply／push、問題前 30 字）。

**停止**：在終端機按 `Ctrl+C`。停止後訪客傳訊息不會有回應；再次執行即可恢復（Webhook 網址每次不同，會自動更新，不需到後台設定）。

### 7.2 常見狀況

| 畫面訊息 | 原因與處理 |
|---|---|
| `No module named rag_chatbot` | 用到系統的 Python；指令開頭須為 `.venv\Scripts\python.exe` |
| `.env 缺少 LINE_CHANNEL_SECRET…` | `.env` 不存在、檔名不對，或內容少一行（見上方「第一次設定」） |
| `無法連線到 Ollama` | Ollama 未執行：從「開始」選單開啟 Ollama 後重試 |
| `找不到 cloudflared` | 執行 `winget install Cloudflare.cloudflared` |
| `通道網址在 120 秒內無法連線`、`無法建立 Cloudflare 通道` | 網路不穩或 Cloudflare 暫時異常；確認網路後重試 |
| `LINE API 錯誤 401` | access token 錯誤或已重新發行：更新 `.env` 後重試 |
| `連線測試：失敗` | 新網址尚未生效；通常稍後即正常，可傳訊息測試，仍無回應則重新啟動 |
| 「注意：LINE 的『Use webhook』尚未開啟」 | 到 Developers Console 打開「Use webhook」（只需一次） |
| `Cloudflare 通道已中斷，請重新啟動。` | 通道斷線（網路中斷、筆電睡眠）；重新執行啟動指令 |

### 7.3 示範期間注意

- Windows 電源設定改為「接上電源時不進入睡眠」；闔上螢幕不睡眠（設定 → 系統 → 電源）。
- 終端機視窗不要關閉；可以縮小。
- 回答時間約 15～25 秒；同時有多人提問時依序回答，後面的人會等比較久。
- 訪客問答與命令列相同，保存在對話紀錄（情境 `intro`）。

## 8. 調整拒答門檻

`min_score`（預設 0.40）：問題與最相關段落的語意分數（0～1）低於此值時，直接回拒答文字、不呼叫 LLM。它只負責擋掉明顯無關的問題；其餘能否回答由 LLM 判斷（D-022）。

**何時要調整**

| 現象 | 代表 | 調整 |
|---|---|---|
| 文件有答案，卻在幾秒內就回拒答文字（未呼叫 LLM） | 門檻太高，誤擋相關問題 | 調低 |
| 無關問題被送進 LLM，回答答非所問，或等很久才拒答 | 門檻太低 | 調高 |

**檢查時機**：每個里程碑的功能測試時；開放 LINE 試用後，定期檢視對話紀錄中被拒答的問題（`messages.kind`、`messages.top_score`）。

**調整步驟**

1. 收集「應該回答」與「應該拒答」的問題各數題。
2. 以 `.venv\Scripts\python.exe -X utf8 -m rag_chatbot search "問題"` 查看每題第 1 名的分數（`ask`／`chat` 的回答開頭也會顯示；M5 起管理介面也會顯示）。
3. 把門檻設在兩組分數之間：修改 `config.toml` 的 `[retrieval] min_score`；只想調整某個情境時，在 `profiles/<情境>.toml` 加 `min_score = 0.42`。不需改程式、不需重新匯入。
4. 驗證：重跑這些問題（M6 起可用批次評估整批重跑），確認沒有讓其他題目變差。
5. 在 [04 §4](04-testing.md#4-執行紀錄) 記錄調整前後的分數與理由。
