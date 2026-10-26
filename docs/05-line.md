# 05 LINE 串接（M4）

← 回到 [主檔](00-main.md)

> 狀態：設計已確認（10-25，§7 全部採用建議，D-030）；實作完成，待功能測試（FT-23～30）。操作方式見 [03 §7](03-environment.md#7-line-機器人m4)。

## 1. 目標

訪客加入 LINE 官方帳號「蔡佳惠 Janice」為好友後，用手機傳文字問題，由筆電上的 `answer()`（`intro` 情境）回答。管理與測試仍使用命令列（M3）；網頁管理介面於 M5。

## 2. 需求

| 編號 | 需求 | 說明 / 驗收重點 |
|---|---|---|
| L-01 | 文字問答 | 一對一聊天中傳文字，收到依文件內容的回答 |
| L-02 | 等待提示 | 收到問題後立即顯示「輸入中」動畫（最長 60 秒） |
| L-03 | 回答時間 | 一般 ≤ 30 秒；超過約 50 秒時仍要送達（改用推播補送） |
| L-04 | 多輪對話 | 同一位訪客連續追問沿用同一對話；閒置超過 30 分鐘後開始新對話 |
| L-05 | 加好友歡迎 | 加入好友時回覆情境的 `greeting_text` |
| L-06 | 非文字訊息 | 貼圖、圖片、語音等回覆「目前只能回答文字問題」 |
| L-07 | 手機易讀 | 回答移除 `[1]`、`[2]` 等引用標記（訪客看不到參考資料）；LINE 不支援 Markdown，送出前整理為純文字：去除 `**`、`#`、`` ` `` 等符號，標題改為「【標題】」，條列改用「•」，長段落依句號拆成約 70 字一段，段落與條列間空一行，「…如下：」直接接條列 |
| L-08 | 錯誤處理 | Ollama 未啟動或回答失敗時，回覆「系統暫時無法回答，請稍後再試，或來信 {contact}」 |
| L-09 | 濫用防護 | 每位訪客每 10 分鐘最多 10 題；等待中的問題超過 3 則時，新問題回覆「目前詢問的人較多，請稍候再問」 |
| L-10 | 安全 | 驗證 LINE 簽章，拒絕偽造請求；金鑰只存在 `.env`；管理介面與命令列不對外 |
| L-11 | 紀錄 | 問答寫入對話紀錄（情境 `intro`），與命令列相同，可用來調整門檻與 Prompt |
| L-12 | 啟動簡單 | 一個指令啟動 webhook、HTTPS 通道，並自動更新 LINE 的 Webhook 網址 |

範圍外：群組／多人聊天（不回應）、圖片內容辨識、主動推播行銷訊息、Rich Menu。

## 3. 架構

```
訪客手機 ──LINE──▶ LINE Platform ──HTTPS──▶ Cloudflare 通道 ──▶ 筆電 127.0.0.1:8000
                                                                    │
                                              FastAPI /callback：驗證簽章 → 排入佇列 → 立即回 200
                                                                    │
                                              背景工作執行緒（一次處理一題）
                                                ├─ 顯示「輸入中」動畫（60 秒）
                                                ├─ RagService.answer(問題, "intro", 該訪客的對話)
                                                ├─ 整理回答（移除 [n]）
                                                └─ reply（50 秒內）；逾時或 token 失效 → push 補送
```

- 回答在背景執行：LINE 要求 webhook 很快回應，回答需 15～25 秒，不能等回答完才回應。
- 一次只處理一題：3B 模型在 CPU 上同時跑兩題只會一起變慢。
- 程式只聽 `127.0.0.1`，外部只能經由通道到達 `/callback`。

## 4. 主要流程

### 4.1 接收（`/callback`）
1. 以 Channel secret 驗證 `X-Line-Signature`（HMAC-SHA256）；不符回 400。
2. 依 `webhookEventId` 去除重複事件（LINE 可能重送）。
3. 只處理一對一聊天（`source.type == "user"`）；群組事件忽略。
4. 事件排入佇列，立即回 200。

### 4.2 處理（背景執行緒）
| 事件 | 處理 |
|---|---|
| 加好友（follow） | reply `greeting_text` |
| 文字訊息 | 檢查頻率限制與佇列長度 → 顯示動畫 → `answer()` → 整理 → reply／push |
| 其他訊息（貼圖、圖片…） | reply「目前只能回答文字問題，歡迎直接輸入您想了解的內容。」 |
| 封鎖（unfollow）等 | 忽略 |

- 問候、道謝等（M3 規則判斷）0 秒回覆，不顯示動畫。
- 回覆時間：從收到事件起算超過 `reply_deadline_seconds`（50 秒）就改用 push；reply 失敗（token 失效）也改用 push。push 計入每月免費 200 則額度，預期很少發生。

### 4.3 訪客與對話
- 新資料表 `line_users(user_id PK, conversation_id, last_active_at)`（schema v3）。
- 距上次發問超過 `conversation_idle_minutes`（30 分鐘）→ 建立新對話。
- 只保存 LINE 的 userId（不取得暱稱、頭像），對話標題為第一個問題。

### 4.4 啟動與通道
```powershell
python -X utf8 -m rag_chatbot line          # 啟動 LINE 機器人，Ctrl+C 停止
```
1. 讀取 `.env` 的金鑰，檢查 Ollama 與模型。
2. 啟動 webhook（`127.0.0.1:8000`）。
3. 啟動 Cloudflare 快速通道（`cloudflared tunnel --url http://127.0.0.1:8000`），取得 `https://xxxx.trycloudflare.com`。
4. 呼叫 LINE API 把 Webhook 網址設為 `https://xxxx.trycloudflare.com/callback`，並執行 LINE 的連線測試（新網址生效需數秒，失敗時自動重試）。同時讀取「Use webhook」開關狀態，未開啟時提示使用者。
5. 顯示「LINE 機器人已上線」，之後每則問答在畫面上列一行（時間、類型、耗時）。

快速通道每次啟動網址都不同，由步驟 4 自動更新，使用者不需手動到後台修改。

- **第一次啟動後**，需到 LINE Developers Console →「Messaging API」→「Webhook settings」打開「Use webhook」（只需一次；之後網址變更不影響此開關）。官方帳號後台「回應設定」的 Webhook 開關即同一設定。
- `cloudflared` 位置：先找 PATH，再找 `C:\Program Files (x86)\cloudflared\cloudflared.exe`；也可在 `[line] cloudflared_path` 指定。

## 5. 設定

`config.toml` 新增：
```toml
[line]
profile = "intro"                 # 使用的情境
port = 8000
reply_deadline_seconds = 50       # 超過改用 push
conversation_idle_minutes = 30    # 閒置多久開始新對話
rate_limit_count = 10             # 每位訪客在 rate_limit_minutes 內最多幾題
rate_limit_minutes = 10
max_queue = 3                     # 等待中的問題上限
cloudflared_path = ""             # 空白表示自動尋找
```

`.env`（不進 Git，使用者自行填寫）：
```
LINE_CHANNEL_SECRET=...
LINE_CHANNEL_ACCESS_TOKEN=...
```

## 6. 模組與測試

| 檔案 | 內容 |
|---|---|
| `line/signature.py` | 簽章驗證 |
| `line/api.py` | LINE API：reply、push、動畫、設定 Webhook 網址（httpx；以介面抽象化，測試用假物件） |
| `line/handler.py` | 事件處理：去重、頻率限制、佇列、回答整理、reply／push 切換 |
| `line/users.py` | 訪客與對話對應 |
| `line/app.py` | FastAPI `/callback` |
| `line/runner.py` | 啟動 webhook、通道、更新 Webhook 網址 |
| `env.py` | 讀取 `.env` |

- 新增依賴：`fastapi`、`uvicorn`。不使用 `line-bot-sdk`：需要的 API 只有 5 個，自行實作較精簡、好測試。
- 外部工具：`cloudflared`（`winget install Cloudflare.cloudflared`）。
- 單元測試全部使用假的 LINE API 與假 LLM，不需網路。

**功能測試（以手機實測）**

| 編號 | 案例 | 預期結果 |
|---|---|---|
| FT-23 | 加好友 | 收到問候文字 |
| FT-24 | 問「她管理過多大的團隊？」 | 先看到「輸入中」，30 秒內收到回答，無 `[1]` 標記 |
| FT-25 | 追問「那她怎麼帶團隊？」 | 回答延續前一題脈絡 |
| FT-26 | 傳貼圖 | 收到「目前只能回答文字問題…」 |
| FT-27 | 兩支手機幾乎同時提問 | 兩人都收到回答（第二位較慢，可能經由 push） |
| FT-28 | 關閉 Ollama 後提問 | 收到「系統暫時無法回答…」 |
| FT-29 | 1 分鐘內連續傳 11 題 | 第 11 題收到頻率限制提示 |
| FT-30 | 以錯誤簽章送出請求 | 回應 400，不處理 |

## 7. 設計選擇（已確認）

| 項目 | 建議 | 其他選擇 |
|---|---|---|
| A. HTTPS 通道 | **Cloudflare 快速通道**：免註冊；網址每次不同，程式自動更新到 LINE | ngrok：需註冊免費帳號，可取得固定網址；Cloudflare 具名通道：需有自己的網域 |
| B. 回答逾時 | **改用 push 補送**（佔免費額度，預期很少） | 不補送，訪客收不到回答 |
| C. 新對話時機 | **閒置 30 分鐘** | 每題獨立（無追問）；永遠同一對話 |
| D. 濫用防護 | **每人 10 分鐘 10 題、佇列上限 3** | 不限制 |
| E. 群組 | **不回應** | 被 @ 時回答 |
| F. 加好友歡迎訊息 | **由程式回覆 `greeting_text`**；LINE 後台內建的「加入好友的歡迎訊息」關閉，避免重複 | 用 LINE 後台內建的歡迎訊息 |

## 8. 使用者需完成的準備

- [x] 再次重新產生 Channel secret
- [x] 發行 Channel access token（long-lived）
- [ ] 官方帳號後台「回應設定」：聊天關閉、自動回應訊息關閉、「加入好友的歡迎訊息」關閉（Webhook 開關需先有網址，第一次啟動後再開）
- [x] 安裝 cloudflared（2026.10.3）
- [ ] 依 [03 §7](03-environment.md#7-line-機器人m4) 建立 `.env`（不要貼到對話中），啟動後開啟「Use webhook」
