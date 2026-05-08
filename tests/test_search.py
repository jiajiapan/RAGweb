import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

os.environ.setdefault("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2")
os.environ.setdefault("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class TestRetrieve:
    def test_retrieve_no_rerank(self):
        with (
            patch("sentence_transformers.SentenceTransformer"),
            patch("sentence_transformers.CrossEncoder"),
        ):
            from backend.search import retrieve

        with (
            patch("backend.search.db") as mock_db,
            patch("backend.search.retriever") as mock_retriever,
        ):
            mock_retriever.encode.return_value = [0.1] * 768
            mock_table = MagicMock()
            mock_table.search.return_value.limit.return_value.to_list.return_value = [
                {"text": "chunk A"},
                {"text": "chunk B"},
                {"text": "chunk C"},
            ]
            mock_db.open_table.return_value = mock_table

            results = retrieve("vs_test", "test query", k=3, rerank=False)

            assert results == ["chunk A", "chunk B", "chunk C"]

    def test_retrieve_with_rerank(self):
        with (
            patch("sentence_transformers.SentenceTransformer"),
            patch("sentence_transformers.CrossEncoder"),
        ):
            from backend.search import retrieve

        with (
            patch("backend.search.db") as mock_db,
            patch("backend.search.retriever") as mock_retriever,
            patch("backend.search.reranker") as mock_reranker,
        ):
            mock_retriever.encode.return_value = [0.1] * 768

            df = pd.DataFrame({"text": ["chunk A", "chunk B", "chunk C", "chunk D", "chunk E"]})
            mock_table = MagicMock()
            mock_table.search.return_value.limit.return_value.to_pandas.return_value = df
            mock_db.open_table.return_value = mock_table

            mock_reranker.predict.return_value = [0.9, 0.3, 0.7, 0.5, 0.1]

            results = retrieve("vs_test", "test query", k=5, rerank=True, top_k=3)

            assert len(results) == 3
            assert results == ["chunk A", "chunk C", "chunk D"]

    def test_retrieve_table_error(self):
        with (
            patch("sentence_transformers.SentenceTransformer"),
            patch("sentence_transformers.CrossEncoder"),
        ):
            from backend.search import retrieve

        with (
            patch("backend.search.db") as mock_db,
            patch("backend.search.retriever") as mock_retriever,
        ):
            mock_retriever.encode.return_value = [0.1] * 768
            mock_db.open_table.side_effect = Exception("Table not found")

            with pytest.raises(Exception):  # noqa: B017
                retrieve("vs_missing", "query")


class TestSearch:
    def test_search_returns_results(self):
        with (
            patch("sentence_transformers.SentenceTransformer"),
            patch("sentence_transformers.CrossEncoder"),
        ):
            from backend.search import search

        with (
            patch("backend.search.db") as mock_db,
            patch("backend.search.retriever") as mock_retriever,
        ):
            mock_retriever.encode.return_value = [0.1] * 768
            mock_table = MagicMock()
            mock_db.open_table.return_value = mock_table

            search("vs_test", "query", k=10)

            mock_table.search.assert_called_once()
            call_args = mock_table.search.call_args
            assert call_args[1]["vector_column_name"] == "vector"
