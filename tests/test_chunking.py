from app.core.ingest import _chunk_text, _normalize, _table_to_text


def test_normalize():
    assert _normalize("  Apple   Inc.\n  ") == "Apple Inc."


def test_chunking():
    chunks = _chunk_text(" ".join(["x"] * 250), size=100, overlap=10)
    assert len(chunks) == 3
    assert all(chunks)


def test_table_text():
    table = [["A", "B"], ["1", "2"]]
    assert _table_to_text(table) == "A | B\n1 | 2"
