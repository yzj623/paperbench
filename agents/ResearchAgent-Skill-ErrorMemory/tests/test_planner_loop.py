from research_agent.readonly import SafeWorkspace


def test_read_file_matches_base_ra_decoding(tmp_path):
    submission = tmp_path / 'submission'
    submission.mkdir()
    path = submission / 'image.jpg'
    path.write_bytes(b'\xff\xd8\xff\x00binary')
    workspace = SafeWorkspace(public_root=tmp_path, workspace_root=tmp_path)
    assert 'binary' in workspace.read_file('image.jpg')
    path.write_text('论文\ntext', encoding='utf-8')
    assert '1: 论文\n2: text' in workspace.read_file('image.jpg')
