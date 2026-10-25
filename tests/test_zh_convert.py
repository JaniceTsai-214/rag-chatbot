import pytest

from rag_chatbot.zh_convert import has_simplified, to_traditional


@pytest.mark.parametrize(
    "text, expected",
    [
        # 實測：qwen2.5 夾雜簡體字
        ("根据现有的资料，系統支持与主流ERP系统对接。", "根據現有的資料，系統支持與主流ERP系統對接。"),
        # 只轉含簡體字的句子，其他句子（含繁簡共用字）不變
        ("避免干擾，台灣製造。这个专案以后会持续优化。", "避免干擾，台灣製造。這個專案以後會持續優化。"),
        ("- 干擾減少 [1]\n- 这是简体\n- 皇后與台灣", "- 干擾減少 [1]\n- 這是簡體\n- 皇后與台灣"),
        # 只轉字形、不換用語：專有名詞不變
        ("天行智能的智能排班", "天行智能的智能排班"),
        ("天行智能的智能排班，这很好。", "天行智能的智能排班，這很好。"),
        ("", ""),
        ("English only, emoji 😀", "English only, emoji 😀"),
    ],
)
def test_to_traditional(text, expected):
    assert to_traditional(text) == expected


def test_has_simplified():
    assert has_simplified("这个")
    assert not has_simplified("這個干擾台灣")
    assert not has_simplified("ERP 123")
