import pytest

from research_agent.readonly import SafeWorkspace


def test_binary_is_not_decoded_as_evidence(tmp_path):
    path = tmp_path / 'image.jpg'
    path.write_bytes(b'\xff\xd8\xff\x00binary')
    with pytest.raises(ValueError, match='UTF-8 text only'):
        SafeWorkspace._text_lines(path)
    path.write_text('论文\ntext', encoding='utf-8')
    assert SafeWorkspace._text_lines(path) == ['论文', 'text']
