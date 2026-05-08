from unittest.mock import MagicMock, PropertyMock, patch

import pytest


def _import_query_deepseek():
    """Import query_deepseek with a mocked openai.Client so module-level init doesn't fail."""
    with patch("openai.Client"):
        from backend.query import query_deepseek

        return query_deepseek


@pytest.fixture(scope="module")
def query_deepseek():
    return _import_query_deepseek()


class TestQueryDeepSeek:
    def test_streaming_response(self, query_deepseek):
        mock_chunk_1 = MagicMock()
        type(mock_chunk_1.choices[0].delta).content = PropertyMock(return_value="Hello")
        mock_chunk_2 = MagicMock()
        type(mock_chunk_2.choices[0].delta).content = PropertyMock(return_value=" world")

        with patch("backend.query.DEEPSEEK_CLIENT") as mock_client:
            mock_client.chat.completions.create.return_value = [
                mock_chunk_1,
                mock_chunk_2,
            ]

            results = list(query_deepseek("Test prompt"))

        assert results == ["Hello", "Hello world"]

    def test_empty_stream(self, query_deepseek):
        with patch("backend.query.DEEPSEEK_CLIENT") as mock_client:
            mock_client.chat.completions.create.return_value = []

            results = list(query_deepseek("Test prompt"))

        assert results == []

    def test_none_content_chunks(self, query_deepseek):
        mock_none = MagicMock()
        type(mock_none.choices[0].delta).content = PropertyMock(return_value=None)
        mock_text = MagicMock()
        type(mock_text.choices[0].delta).content = PropertyMock(return_value="ok")

        with patch("backend.query.DEEPSEEK_CLIENT") as mock_client:
            mock_client.chat.completions.create.return_value = [mock_none, mock_text]

            results = list(query_deepseek("Test prompt"))

        assert results == ["ok"]

    def test_too_many_requests_error(self, query_deepseek):
        with patch("backend.query.DEEPSEEK_CLIENT") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("Too Many Requests")

            with pytest.raises(Exception, match="Too many requests on DeepSeek client"):
                list(query_deepseek("prompt"))

    def test_missing_api_key_error(self, query_deepseek):
        with patch("backend.query.DEEPSEEK_CLIENT") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception(
                "You didn't provide an API key"
            )

            with pytest.raises(Exception, match="DeepSeek API key was either not provided"):
                list(query_deepseek("prompt"))
