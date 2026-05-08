import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_text():
    return "This is the first sentence. This is the second sentence. Here is a third sentence. And a fourth one. Fifth sentence here."


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a minimal PDF file for testing pdf_parser."""
    import fitz

    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world. This is page one.")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def multiline_text():
    return (
        "Machine learning is a subset of artificial intelligence. "
        "It enables systems to learn from data. "
        "Deep learning uses neural networks with many layers. "
        "Natural language processing deals with text data. "
        "Computer vision focuses on image understanding. "
        "Reinforcement learning involves agents and environments. "
        "Transfer learning reuses pre-trained models for new tasks. "
        "Supervised learning requires labeled training data."
    )
