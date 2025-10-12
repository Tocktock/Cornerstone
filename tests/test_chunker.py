from __future__ import annotations

from cornerstone.chunker import chunk_markdown


def test_chunk_markdown_produces_metadata_and_overlap():
    repeated_sentence = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    # Ensure multiple chunks by repeating enough sentences under a heading
    paragraph = " ".join(repeated_sentence for _ in range(80))
    text = """# Handbook\n\n## Getting Started\n\n{para}\n\n## Advanced\n\nMore content here.""".format(para=paragraph)

    chunks = chunk_markdown(text, max_tokens=120, overlap_tokens=12, min_tokens=60)

    assert len(chunks) >= 2

    first = chunks[0]
    second = chunks[1]

    assert first.heading_path == ("Handbook", "Getting Started")
    assert first.section_title == "Getting Started"
    assert first.summary is not None
    assert first.language == "en"
    assert first.token_count > 0
    assert first.char_count == len(first.text)

    # Overlap should duplicate some trailing words into the next chunk for continuity
    first_tail = first.text.split()[-10:]
    second_head = second.text.split()[:10]
    assert set(first_tail) & set(second_head)

    # Later section should update heading path
    assert any(
        chunk.heading_path and chunk.heading_path[-1] == "Advanced"
        for chunk in chunks
    )
