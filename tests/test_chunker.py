from src.chunker import chunk_text


def test_chunk_text_returns_chunks():
    text = "A" * 2000

    chunks = chunk_text(
        text,
        chunk_size=800,
        chunk_overlap=120
    )

    assert len(chunks) > 1


def test_chunk_size_is_respected():
    text = "A" * 2000

    chunks = chunk_text(
        text,
        chunk_size=800,
        chunk_overlap=120
    )

    assert all(len(chunk) <= 800 for chunk in chunks)


def test_invalid_overlap_raises_error():
    text = "Example text"

    try:
        chunk_text(
            text,
            chunk_size=100,
            chunk_overlap=100
        )
        assert False
    except ValueError:
        assert True