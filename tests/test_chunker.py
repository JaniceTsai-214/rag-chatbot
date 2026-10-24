import pytest

from rag_chatbot.chunker import chunk_sections
from rag_chatbot.models import Chunk, Section


def _sentences(count: int, length: int = 20) -> str:
    """產生 count 句、每句 length 字（含句號）的文字，句子內容互不相同。"""
    return "".join(f"{i:03d}" + "字" * (length - 4) + "。" for i in range(count))


def test_short_section_not_split():
    chunks = chunk_sections([Section("  短文。  ", page=2, heading="標題")], 600, 100)
    assert chunks == [Chunk(0, "短文。", page=2, heading="標題")]


def test_exact_size_not_split():
    text = "字" * 600
    assert len(chunk_sections([Section(text)], 600, 100)) == 1


def test_length_limit_and_sentence_breaks():
    text = _sentences(100)  # 2,000 字
    chunks = chunk_sections([Section(text)], 600, 100)
    assert len(chunks) > 3
    assert all(len(c.text) <= 600 for c in chunks)
    # 除了最後一段，都在句號處斷開
    assert all(c.text.endswith("。") for c in chunks[:-1])
    assert chunks[-1].text.endswith(text[-20:])


def test_overlap_starts_at_sentence():
    text = _sentences(100)  # 每句 20 字，重疊 100 字內可容納 4 句完整句子
    chunks = chunk_sections([Section(text)], 600, 100)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.text.startswith(prev.text[-80:])
        assert not nxt.text.startswith(prev.text[-81:])
        assert nxt.text[:3].isdigit()  # 從句首開始


def test_overlap_mid_sentence_when_sentence_too_long():
    text = "字" * 1500
    chunks = chunk_sections([Section(text)], 600, 100)
    assert len(chunks[1].text) == 600
    assert chunks[0].text[-100:] == chunks[1].text[:100]


def test_zero_overlap_covers_text_exactly():
    text = _sentences(100)
    chunks = chunk_sections([Section(text)], 600, 0)
    assert "".join(c.text for c in chunks) == text


def test_prefers_paragraph_over_sentence():
    para1 = _sentences(20)  # 400 字
    para2 = _sentences(20)
    chunks = chunk_sections([Section(para1 + "\n\n" + para2)], 600, 50)
    assert chunks[0].text == para1


def test_break_not_too_early():
    # 空行太靠前（未過半）時不採用，改在後段的句號斷開
    text = "前言。\n\n" + _sentences(50)
    chunks = chunk_sections([Section(text)], 600, 0)
    assert len(chunks[0].text) > 300


def test_comma_break_when_no_sentence_end():
    text = ("一二三四五六七八九，" * 100).rstrip("，")
    chunks = chunk_sections([Section(text)], 100, 0)
    assert all(c.text.endswith("，") for c in chunks[:-1])


def test_hard_cut_without_breaks():
    text = "字" * 1500
    chunks = chunk_sections([Section(text)], 600, 100)
    assert [len(c.text) for c in chunks] == [600, 600, 500]


def test_english_sentence_break():
    text = "This is a sentence. " * 50
    chunks = chunk_sections([Section(text)], 200, 0)
    assert all(c.text.endswith(".") for c in chunks)


def test_does_not_cross_sections_and_keeps_location():
    sections = [
        Section(_sentences(50), page=1),
        Section("第二頁短文。", page=2),
        Section(_sentences(50), heading="章節"),
    ]
    chunks = chunk_sections(sections, 600, 100)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks if "第二頁" in c.text} == {2}
    page1 = [c for c in chunks if c.page == 1]
    assert "第二頁" not in "".join(c.text for c in page1)
    assert all(c.heading == "章節" for c in chunks if c.page is None)


def test_no_split_mode_for_faq():
    long_answer = "問：問題\n答：" + "字" * 1000
    chunks = chunk_sections([Section(long_answer, heading="分類"), Section("問：b\n答：c")], 600, 100, split=False)
    assert chunks == [Chunk(0, long_answer, heading="分類"), Chunk(1, "問：b\n答：c")]


def test_empty_sections_skipped():
    chunks = chunk_sections([Section("   "), Section("內容")], 600, 100)
    assert chunks == [Chunk(0, "內容")]


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_sections([Section("x")], size, overlap)


def test_large_overlap_still_progresses():
    text = "字" * 1000
    chunks = chunk_sections([Section(text)], 100, 99)
    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[-1].text.endswith("字")
