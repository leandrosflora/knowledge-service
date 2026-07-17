from app.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("   ", chunk_size=100, chunk_overlap=10) == []
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_short_text_produces_a_single_chunk():
    text = "Como funciona a renegociacao?"
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == text


def test_long_text_produces_overlapping_chunks_with_sequential_indexes():
    text = "a" * 2500
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=150)

    # step = 1000 - 150 = 850; starts at 0, 850, 1700 -> 3 windows covering 2500 chars
    assert [c.index for c in chunks] == [0, 1, 2]
    assert all(len(c.text) <= 1000 for c in chunks)
    # consecutive chunks overlap: the tail of chunk 0 reappears at the head of chunk 1
    assert chunks[0].text[-100:] == chunks[1].text[:100]


def test_chunk_overlap_must_be_smaller_than_chunk_size_to_progress():
    text = "b" * 3000
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)

    # step = 400, so we should get multiple, strictly-increasing-start chunks
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
