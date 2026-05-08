import os

import gradio as gr
import lancedb
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch.nn import Sigmoid

db = lancedb.connect(".lancedb")

BATCH_SIZE = 32
VEC_COLUMN = "vector"  # name of vector column is "vector"
TEXT_COLUMN = "text"  # name of text column is "text"

retriever = SentenceTransformer(os.getenv("EMB_MODEL"), device="cpu")  # get an embedding model
reranker = CrossEncoder(
    os.getenv("RERANK_MODEL"), device="cpu"
)  # get an rerank model, which will give a score to (query, chunk)


def search(vs_name, query, k=25):  # search function can saerch the 25 closest rows in table vs_name
    query_vec = retriever.encode(query, show_progress_bar=False)  # turn query to vec
    try:
        table = db.open_table(vs_name)
        search_results = table.search(query_vec, vector_column_name=VEC_COLUMN).limit(k)
    except Exception as e:
        raise gr.Error(str(e)) from e
    return search_results


def retrieve(
    vs_name, query, k=25, rerank=False, top_k=5
):  # return the actual text, if not rerank, just k results, if rerank, top_k results
    try:
        semantic_search = search(vs_name, query, k=k)  # get the search results using search method
        # why use rerank? using retriever, we can get _distance, but it is only spatial distance for embeddings.
        # using reranker, we can get the semantic distance, which is better.
        # but retriever is quick to get the results for searching.
        # Sigmoid, turn [-..., ...] to [0,1]
        if rerank:
            chunks = semantic_search.to_pandas().reset_index(
                drop=True
            )  # turn results into pandas dataframe, where old index is lost and 0,1,2... new index is set
            query_chunk_comb = [
                [query, chunk] for chunk in chunks[TEXT_COLUMN]
            ]  # [[query, chunk1], [query, chunk2], ...]
            chunks["_distance_reranked"] = reranker.predict(
                query_chunk_comb, activation_fct=Sigmoid()
            )
            chunks = chunks.sort_values("_distance_reranked", ascending=False).head(top_k)
            retrievals = list(chunks.text)
        else:
            retrievals = [doc[TEXT_COLUMN] for doc in semantic_search.to_list()]
    except Exception as e:
        raise gr.Error(str(e)) from e
    return retrievals
