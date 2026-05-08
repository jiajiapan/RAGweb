from backend.propressing import chunk_text, pdf_parser


class TestChunkText:
    def test_basic_chunking(self, sample_text):
        chunks = chunk_text(sample_text, num_sents=2)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "sentence_chunk" in chunk
            assert "chunk_char_count" in chunk
            assert "chunk_word_count" in chunk
            assert isinstance(chunk["sentence_chunk"], str)
            assert isinstance(chunk["chunk_char_count"], int)
            assert isinstance(chunk["chunk_word_count"], int)

    def test_num_sents_controls_chunk_size(self, multiline_text):
        chunks_2 = chunk_text(multiline_text, num_sents=2)
        chunks_4 = chunk_text(multiline_text, num_sents=4)
        assert len(chunks_2) > len(chunks_4)

    def test_single_sentence_input(self):
        chunks = chunk_text("Just one sentence.", num_sents=4)
        assert len(chunks) == 1
        assert chunks[0]["sentence_chunk"] == "Just one sentence."

    def test_empty_string(self):
        chunks = chunk_text("", num_sents=4)
        assert len(chunks) == 0

    def test_output_text_is_non_empty(self, sample_text):
        chunks = chunk_text(sample_text, num_sents=3)
        for chunk in chunks:
            assert len(chunk["sentence_chunk"]) > 0

    def test_char_count_matches_text_length(self, sample_text):
        chunks = chunk_text(sample_text, num_sents=2)
        for chunk in chunks:
            assert chunk["chunk_char_count"] == len(chunk["sentence_chunk"])

    def test_unsupported_method_raises(self, sample_text):
        import pytest

        with pytest.raises(ValueError, match="Unsupported chunking method"):
            chunk_text(sample_text, method="paragraph")

    def test_chunks_not_empty_for_short_text(self):
        chunks = chunk_text("Short.", num_sents=10)
        assert len(chunks) == 1


class TestPdfParser:
    def test_returns_list_of_pages(self, sample_pdf_path):
        pages = pdf_parser(sample_pdf_path)
        assert isinstance(pages, list)
        assert len(pages) == 1

    def test_page_has_expected_keys(self, sample_pdf_path):
        pages = pdf_parser(sample_pdf_path)
        page = pages[0]
        assert page["page"] == 1
        assert "page_chars" in page
        assert "page_words" in page
        assert "text" in page

    def test_page_text_contains_content(self, sample_pdf_path):
        pages = pdf_parser(sample_pdf_path)
        assert "Hello world" in pages[0]["text"]

    def test_char_count_positive(self, sample_pdf_path):
        pages = pdf_parser(sample_pdf_path)
        assert pages[0]["page_chars"] > 0
        assert pages[0]["page_words"] > 0
