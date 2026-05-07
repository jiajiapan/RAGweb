import fitz #fitz can read text from pdf
from tqdm.auto import tqdm #tqdm is a processing bar tool
from spacy.lang.en import English #use spaCy lib to complete nlp 
import re #
import os # get env variables
from sentence_transformers import SentenceTransformer
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import List, Dict
import hashlib


# This function can transform a pdf file to a list.
# This list, contains information of every page. For one page, it has key words, "page", "page_chars", "page_words", "text".
def pdf_parser(pdf_path:str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    
    for i, page in tqdm(enumerate(doc)):
        text = page.get_text()
        pages.append({
            "page": i + 1,
            "page_chars": len(text),
            "page_words": len(text.split(" ")),
            "text": text,
        })

    return pages

# This function can chunk the text. The chunking strategy is segmenting text into fixed-size chunks. The default arg is 4.
# The list, contains information of every chunk. For one chunk, it has key words, "sentence_chunk", "chunk_char_count", "chunk_word_count".
# Here is space for other chunk methods. You can change the arg method to achieve other chunk methods.
def chunk_text(text, num_sents: int = 4, method: str = "sentence") -> List[Dict]:
    if method == "sentence":
        # This returns the sentence list of the input text.
        nlp = English()
        nlp.add_pipe("sentencizer")
        dot_sep_list = list(nlp(text).sents) 
        dot_sep_str_list = [str(x) for x in dot_sep_list]
        
        # This returns chunk list (list of list), and every chunk contains a list.
        sentence_chunks = [
            dot_sep_str_list[i : i + num_sents] for i in range(0, len(dot_sep_list), num_sents)
        ]

        text_chunks = [] # this is a list of dict, a dict contains information of every chunk.
        for chunk in sentence_chunks:
            chunkd = {} # this is a dict, the key words are "sentence_chunk", "chunk_char_count", "chunk_word_count".
            joined_chunk = "".join(chunk).replace("  ", " ").strip()
            joined_chunk = re.sub(
                r"\.([A-Z])", r". \1", joined_chunk
            )
            chunkd["sentence_chunk"] = joined_chunk
            chunkd["chunk_char_count"] = len(joined_chunk)
            chunkd["chunk_word_count"] = len([word for word in joined_chunk.split(" ")])
            text_chunks.append(chunkd)
    
        return text_chunks
    else:
        raise ValueError(f"Unsupported chunking method: {method}")
    
# using pdf_parser and chunk_text to get the embedding results and store them into lanceDB.
def embedding(pdf_path:str):
    # Settings of embedding paras.
    SENTS_EMBED_MODEL = os.getenv("EMB_MODEL")
    sents_embedder = SentenceTransformer(SENTS_EMBED_MODEL, device = "cpu")
    sents_embedder.eval() # this is the evaluation mode
    BATCH_SIZE = 32
    EMBEDDING_DIM_MODEL = 768 # the dimension is 768
    NUM_PARTITIONS_VEC = 128
    NUM_SUB_VEC = 96
    VEC_COLUMN = "vector" # name of vector column is "vector"
    TEXT_COLUMN = "text" # name of text column is "text"
    assert (EMBEDDING_DIM_MODEL % NUM_SUB_VEC == 0), "Embedding size must be divisible by the num of sub vectors"

    # Setting of lanceDB
    LANCE_DB_LOC = "./.lancedb" # the location of lanceDB
    db = lancedb.connect(LANCE_DB_LOC)
    schema = pa.schema(
        [
            pa.field(VEC_COLUMN, pa.list_(pa.float32(), EMBEDDING_DIM_MODEL)),
            pa.field(TEXT_COLUMN, pa.string()),
        ]
    )
    vs_hash = hashlib.sha256(f"{SENTS_EMBED_MODEL}_{NUM_SUB_VEC}".encode("utf-8")).hexdigest()
    vs_name = f"vs_{vs_hash}"
    tbl = db.create_table(vs_name, schema = schema, mode = "overwrite") # "overwrite" means if there is a table with the same name, replace it
    

    # sentences stores all the chunks using chunk_text method
    sentences = []
    pdf_page_texts = pdf_parser(pdf_path)
    for page in tqdm(pdf_page_texts):
        # "sentence_chunks" stores chunking results of this page
        page["sentence_chunks"] = chunk_text(page["text"], num_sents=4, method="sentence")
        for chunk in page["sentence_chunks"]:
            sentences.append(chunk["sentence_chunk"])
    
    # embedding
    for i in tqdm(range(0, int(np.ceil(len(sentences)/BATCH_SIZE)))):
        try:
            batch = [sent for sent in sentences[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] if len(sent) > 0]
            encoded = sents_embedder.encode(batch, normalize_embeddings = True)
            encoded = [list(vec) for vec in encoded]
            df = pd.DataFrame({VEC_COLUMN: encoded, TEXT_COLUMN: batch})
            tbl.add(df)
        except KeyboardInterrupt: #this means "ctrl + C", raise means keeping this exception and ending the task
            raise
        except Exception as e:
            print(f"Error on batch #{i}: {e}")
    
    tbl.create_index(
        num_partitions = NUM_PARTITIONS_VEC,
        num_sub_vectors = NUM_SUB_VEC,
        vector_column_name = VEC_COLUMN
    )

    return vs_name # return the name of current table(a vector store)