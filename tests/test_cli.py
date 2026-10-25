import pytest

from rag_chatbot.cli import main
from rag_chatbot.embedder import EmbedderError
from tests.fakes import FakeEmbedder


@pytest.fixture
def run(project, tmp_path, capsys):
    """執行 CLI，回傳 (結束碼, stdout, stderr)。預設使用 FakeEmbedder。"""

    def _run(*args, embedder=None):
        embedder = embedder or FakeEmbedder(model=project.ollama.embed_model)
        code = main(["--config", str(tmp_path / "config.toml"), *args], lambda config: embedder)
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
    assert lines[1].startswith("1. ") and "product/手冊.md [方案]" in lines[1]
    assert lines[2].strip() == "進階版價格每月 60 元。"
    assert "← 低於門檻" in lines[3]  # 履歷與問題無共同關鍵字

    _, out, _ = run("search", "進階版價格", "--profile", "me")
    assert "手冊" not in out and "me/履歷.md" in out


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

    code, _, err = run("search", "x", embedder=Broken(model="bge-m3"))
    assert code == 1 and "錯誤：無法連線到 Ollama" in err
