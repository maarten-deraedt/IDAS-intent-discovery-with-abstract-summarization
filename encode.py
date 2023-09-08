from sentence_transformers import SentenceTransformer
from typing import List

import time


def sentence_transformer(sentences: List[str], model_name: str):
    model = SentenceTransformer(model_name)
    start = time.time()
    embeddings = model.encode(sentences)
    print(f"Took {time.time() - start} seconds to encode {len(sentences)} sentences!")
    return embeddings