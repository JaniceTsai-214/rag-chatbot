import pytest

from rag_chatbot.cli import main
from rag_chatbot.embedder import EmbedderError
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture
def run(project, tmp_path, capsys):
    """執行 CLI，回傳 (結束碼, stdout, stderr)。預設使用 FakeEmbedder 與 FakeLLM。"""

    def _run(*args, embedder=None, llm=None):
        embedder = embedder or FakeEmbedder(model=project.ollama.embed_model)
        llm = llm or FakeLLM("進階版每月 60 元 [1]。")
        code = main(["--config", str(tmp_path / "config.toml"), *args], lambda config: embedder, lambda config: llm)
        out, err = capsys.readouterr()
        return code, out, err

    return _run


@pytest.fixture
def docs(project):
    root = project.paths.collections
    (root / "me" / "履歷.md").write_text("## 工作經歷\n在小算盤團隊工作。", encoding="utf-8")
    (root / "product" / "手冊.md").write_text("## 方案\n進階版價格每月 60 元。", encoding="utf-8")
    (root / "product" / "notes.txt").write_text("x", encoding="utf-8")


def test_profiles(run, project):
    (project.paths.collections / "product").rmdir()
    code, out, _ = run("profiles")
    assert code == 0
    assert "intro（預設）：介紹助理，介紹對象 王小明" in out
    assert "  - me\n" in out
    assert "  - product（資料夾不存在）" in out


def test_profiles_empty(run, project):
    for p in project.paths.profiles.glob("*.toml"):
        p.unlink()
    code, out, _ = run("profiles")
    assert code == 0 and "沒有情境設定" in out


def test_ingest_and_list(run, docs):
    code, out, _ = run("ingest")
    assert code == 0
    assert "[新增] me/履歷.md（1 段）" in out
    assert "[新增] product/手冊.md（1 段）" in out
    assert "[略過] product/notes.txt：不支援的格式" in out
    assert "完成：新增 2、已取代 0、已是最新 0、失敗 0、略過 1" in out

    code, out, _ = run("ingest", "me")
    assert "[已是最新] me/履歷.md（1 段）" in out
    assert "product" not in out

    code, out, _ = run("list")
    assert code == 0
    assert "me/履歷.md  markdown  1 段" in out
    assert "共 2 份文件、2 段" in out
    code, out, _ = run("list", "product")
    assert "履歷" not in out and "共 1 份文件" in out


def test_ingest_failure_exit_code(run, project):
    (project.paths.collections / "me" / "bad.pdf").write_bytes(b"x")
    code, out, _ = run("ingest")
    assert code == 1
    assert "[失敗] me/bad.pdf：bad.pdf：無法開啟 PDF" in out


def test_ingest_no_collections(run, project):
    for name in ("me", "product"):
        (project.paths.collections / name).rmdir()
    code, out, _ = run("ingest")
    assert code == 0 and "沒有文件集" in out


def test_ingest_unknown_collection(run):
    code, _, err = run("ingest", "nope")
    assert code == 1 and "錯誤：找不到文件集資料夾" in err


def test_ingest_rebuild_after_model_change(run, project, docs, tmp_path):
    run("ingest")
    (tmp_path / "config.toml").write_text('[ollama]\nembed_model = "new-model"\n', encoding="utf-8")
    new = FakeEmbedder(model="new-model")

    code, out, _ = run("list", embedder=new)
    assert "注意：資料庫的向量由 bge-m3 產生" in out
    code, _, err = run("ingest", embedder=new)
    assert code == 1 and "ingest --rebuild" in err

    code, out, _ = run("ingest", "--rebuild", embedder=new)
    assert code == 0
    assert "已清空資料庫中的 2 份文件" in out
    assert "新增 2" in out


def test_list_empty_and_missing_original(run, project, docs):
    code, out, _ = run("list")
    assert "沒有已匯入的文件" in out
    run("ingest")
    (project.paths.collections / "me" / "履歷.md").unlink()
    _, out, _ = run("list")
    assert "me/履歷.md  markdown  1 段" in out and "（原始檔不存在）" in out


def test_search(run, docs):
    run("ingest")
    code, out, _ = run("search", "進階版價格", "--top-k", "2")
    assert code == 0
    assert "情境 intro（文件集：me, product），門檻 0.40" in out
    lines = out.splitlines()
    assert lines[1].startswith("語意最高分 ") and lines[1].endswith("→ 送給模型回答")
    assert lines[2].startswith("1. ") and "product/手冊.md [方案]" in lines[2]
    assert lines[3].strip() == "進階版價格每月 60 元。"
    assert lines[4].startswith("2. ") and "me/履歷.md" in lines[4]

    _, out, _ = run("search", "進階版價格", "--profile", "me")
    assert "手冊" not in out and "me/履歷.md" in out
    assert "低於門檻，會直接拒答" in out  # 履歷與問題無共同關鍵字


def test_search_shows_page(run, project):
    from tests.builders import make_pdf

    make_pdf(project.paths.collections / "product" / "手冊.pdf", ["簡介", "方案價格"])
    run("ingest")
    _, out, _ = run("search", "方案價格", "--profile", "intro", "--top-k", "1")
    assert "product/手冊.pdf 第 2 頁" in out


def test_search_edge_cases(run, project):
    code, out, _ = run("search", "問題")
    assert code == 0 and "沒有任何段落" in out
    (project.paths.collections / "product").rmdir()
    _, out, _ = run("search", "問題")
    assert "注意：文件集資料夾不存在：product" in out
    code, _, err = run("search", "問題", "--top-k", "0")
    assert code == 1 and "--top-k" in err
    code, _, err = run("search", "問題", "--profile", "nope")
    assert code == 1 and "找不到情境「nope」" in err


def test_delete(run, project, docs):
    run("ingest")
    code, out, _ = run("delete", "me", "履歷.md")
    assert code == 0 and "已刪除：me/履歷.md（原始檔保留）" in out
    assert (project.paths.collections / "me" / "履歷.md").is_file()
    code, _, err = run("delete", "me", "履歷.md")
    assert code == 1 and "資料庫中沒有 me/履歷.md" in err


def test_embedder_error_message(run, docs):
    failing = FakeEmbedder(model="bge-m3", fail_on_call=1)
    code, _, err = run("ingest", embedder=failing)
    assert code == 1 and "錯誤：FakeEmbedder 模擬失敗" in err


def test_config_error(capsys, tmp_path):
    code = main(["--config", str(tmp_path / "missing.toml"), "list"])
    assert code == 1
    assert "錯誤：找不到設定檔" in capsys.readouterr().err


def test_interactive_progress(run, docs, monkeypatch):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    code, out, _ = run("ingest")
    assert code == 0
    assert "處理中：me/履歷.md" in out
    assert "向量化：1/1 段" in out


def test_module_entry_point():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "rag_chatbot", "--help"], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0
    assert "ingest" in result.stdout


def test_embedder_error_class_is_reported(run, docs):
    class Broken(FakeEmbedder):
        def embed(self, texts, progress=None):
            raise EmbedderError("無法連線到 Ollama")

    run("ingest")
    code, _, err = run("search", "x", embedder=Broken(model="bge-m3"))
    assert code == 1 and "錯誤：無法連線到 Ollama" in err


# ---------- ask / chat ----------

def test_ask(run, docs):
    run("ingest")
    code, out, _ = run("ask", "進階版價格多少？")
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("[回答｜") and "Prompt v1" in lines[0] and "語意最高分" in lines[0]
    assert lines[1] == "進階版每月 60 元 [1]。"
    assert "回答依據：" in out and "product/手冊 > 方案" in out
    assert "（對話 #1；接續追問請加 --conversation 1）" in out

    code, out, _ = run("ask", "那共用呢？", "--conversation", "1")
    assert code == 0 and "對話 #1" in out


def test_ask_refusal_and_greeting(run, docs):
    run("ingest")
    llm = FakeLLM()
    _, out, _ = run("ask", "明天天氣？", llm=llm)
    assert out.startswith("[拒答｜") and "沒有資料，請洽 test@example.com。" in out
    assert "回答依據" not in out and llm.calls == []
    _, out, _ = run("ask", "你好")
    assert out.startswith("[問候｜") and "語意最高分" not in out


def test_ask_errors(run, docs):
    run("ingest")
    code, _, err = run("ask", "   ")
    assert code == 1 and "問題不可為空白" in err
    code, _, err = run("ask", "進階版價格", "--conversation", "99")
    assert code == 1 and "對話 99 不存在" in err
    code, _, err = run("ask", "進階版價格", "--prompt-version", "9")
    assert code == 1 and "沒有 Prompt v9" in err
    code, _, err = run("ask", "進階版價格", llm=FakeLLM(fail=True))
    assert code == 1 and "錯誤：FakeLLM 模擬失敗" in err


def test_chat_session(run, docs, monkeypatch):
    run("ingest")
    inputs = iter(["你好", "進階版價格", "", "不會讀到這裡"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    code, out, _ = run("chat", "--sources")
    assert code == 0
    assert out.startswith("情境 intro：介紹助理。")
    assert "[問候｜" in out and "[回答｜" in out and "回答依據：" in out
    assert out.rstrip().endswith("對話已保存（#1）")


def test_chat_quit_words_and_eof(run, docs, monkeypatch):
    for ending in (["exit"], ["離開"]):
        inputs = iter(ending)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        code, out, _ = run("chat")
        assert code == 0 and "對話已保存" not in out

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert run("chat")[0] == 0


def test_chat_keeps_going_after_llm_error(run, docs, monkeypatch):
    run("ingest")
    inputs = iter(["進階版價格", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    code, _, err = run("chat", llm=FakeLLM(fail=True))
    assert code == 0 and "錯誤：FakeLLM 模擬失敗" in err
