"""產生功能測試用文件（虛構人物「王小明」與虛構產品「小算盤記帳 App」，見 docs/04-testing.md §2.1）。

python tests/functional/make_testdata.py [--force]

已存在的檔案預設不覆寫（DOCX 每次產生的內容 hash 都不同）；修改內容後加 --force 重新產生。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from tests.builders import make_csv, make_docx, make_pdf  # noqa: E402

RESUME = [
    ("para", "王小明，台北人，軟體工程師，擅長後端開發與資料處理。"),
    ("heading", 1, "工作經歷"),
    ("para", "2021 年至今：在小算盤團隊擔任後端工程師，負責雲端同步功能與帳本共用功能的開發。"),
    ("para", "2018 年至 2021 年：在電商公司擔任資料工程師，負責訂單資料分析與報表自動化。"),
    ("table", [
        ["期間", "公司", "職稱"],
        ["2021 年至今", "小算盤團隊", "後端工程師"],
        ["2018–2021 年", "電商公司", "資料工程師"],
    ]),
    ("heading", 1, "專長"),
    ("para", "Python、SQL、雲端服務（AWS）、資料庫設計。"),
    ("para", "擅長把複雜的需求拆解成簡單、容易維護的系統。"),
]

PORTFOLIO = """# 作品集

## 小算盤記帳 App

王小明參與開發的記帳 App，主要負責後端與雲端同步。

- 上架於 iOS 與 Android
- 累積下載超過 10 萬次

## 個人部落格

王小明經營的技術部落格，分享 Python 與資料庫設計的文章，每月更新兩篇。
"""

MANUAL_PAGES = [
    "小算盤記帳 App 產品手冊\n\n產品簡介\n"
    "小算盤是一款簡單好用的記帳 App，幫助個人與家庭輕鬆管理日常收支。"
    "只要三秒鐘就能記下一筆帳，並自動產生每月收支圖表。",

    "方案價格\n\n"
    "免費版：基本記帳、每月收支圖表，最多建立 1 個帳本。\n"
    "進階版：每月 60 元（年繳 600 元），可建立無限個帳本、帳本共用（最多 5 人）、匯出 Excel 報表。\n"
    "學生方案：憑學生證享進階版 5 折優惠，每月 30 元。",

    "支援平台與資料安全\n\n"
    "支援 iOS 15 以上與 Android 10 以上。\n"
    "每日自動備份到雲端，換手機時登入帳號即可還原所有資料。\n"
    "資料傳輸全程加密。",
]

FAQ = [
    ["question", "answer", "category"],
    ["可以多人共用帳本嗎？", "進階版支援帳本共用，最多 5 人。", "功能"],
    ["資料會不會不見？", "每日自動備份到雲端，換手機時登入帳號即可還原。", "資料安全"],
    ["可以匯出報表嗎？", "進階版可以匯出 Excel 報表。", "功能"],
    ["如何取消訂閱？", "在 App 的「設定 > 訂閱管理」中即可取消，取消後可使用到當期結束。", "付款"],
    ["有提供客服嗎？", "可以在 App 內的「意見回饋」留言，客服會在 2 個工作天內回覆。", "服務"],
]


def main(force: bool) -> None:
    me = HERE / "collections" / "me"
    product = HERE / "collections" / "product"
    me.mkdir(parents=True, exist_ok=True)
    product.mkdir(parents=True, exist_ok=True)

    builders = [
        (me / "王小明履歷.docx", lambda p: make_docx(p, RESUME)),
        (me / "作品集.md", lambda p: p.write_text(PORTFOLIO, encoding="utf-8", newline="\n")),
        (product / "小算盤產品手冊.pdf", lambda p: make_pdf(p, MANUAL_PAGES)),
        (product / "小算盤FAQ.csv", lambda p: make_csv(p, FAQ)),
    ]
    for path, build in builders:
        if path.exists() and not force:
            print(f"已存在，略過：{path.relative_to(HERE)}")
            continue
        build(path)
        print(f"已產生：{path.relative_to(HERE)}")


if __name__ == "__main__":
    main(force="--force" in sys.argv[1:])
