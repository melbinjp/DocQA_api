import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import datetime

class RAGSession:
    """
    Manages the RAG process for a single, isolated user session in memory.

    Each instance of this class handles the data for one ingested document,
    including its text chunks, embeddings, and a FAISS index for searching.
    It also tracks its last access time for automatic cleanup.
    """
    def __init__(self, source: str, embedding_model):
        self.source = source
        self.last_accessed = datetime.datetime.now()
        self.embedding_model = embedding_model

        d_model = self.embedding_model.get_sentence_embedding_dimension()

        # Inner product over L2-normalised vectors, which is cosine similarity.
        #
        # This was IndexFlatL2 with unnormalised vectors, and the score reported
        # to the user was `1 / (1 + l2_distance)`. That is not a similarity: it
        # is an unbounded distance squashed into (0, 1] with no meaning at
        # either end. Measured on the live Space, the five correctly retrieved
        # passages for a question about multi-head attention scored 0.081 down
        # to 0.076, and the UI rendered them as "Confidence: 8.1%" beside
        # answers that were right. Cosine puts the same passages near 1.
        #
        # It also ranks better. Untitled L2 distance is sensitive to vector
        # magnitude, and magnitude tracks chunk length more than relevance, so
        # long chunks were being penalised for being long.
        self.index = faiss.IndexFlatIP(d_model)

        # In-memory store for the actual text chunks corresponding to the vectors.
        # The index in this list is the ID used in the FAISS index.
        self.chunks = []

        # Parallel to self.chunks: the page each chunk came from, or None for
        # formats without pages. Kept as a separate list rather than a dict so a
        # FAISS vector id indexes both without a lookup that could disagree.
        self.pages = []

    def ingest(self, text_chunks: list[str], embeddings: np.ndarray, pages: list | None = None):
        """
        Processes and ingests text chunks and their pre-computed embeddings
        into the session's RAG store.

        Args:
            text_chunks: A list of strings, where each string is a chunk of the
                         source document.
            embeddings: A numpy array of the embeddings for the text chunks.
        """
        if not text_chunks:
            return

        # FAISS requires a flat numpy array of float32. Normalising in place is
        # what turns the inner-product index into a cosine index; skip it and
        # every score is meaningless and the ranking is magnitude-biased.
        embeddings_float32 = np.ascontiguousarray(np.array(embeddings, dtype='float32'))
        faiss.normalize_L2(embeddings_float32)

        # Add the new embeddings to the FAISS index.
        self.index.add(embeddings_float32)

        # Store the corresponding text chunks, and the page each came from. The
        # two lists must stay the same length or a citation would name the wrong
        # page, which is worse than naming none.
        self.chunks.extend(text_chunks)
        if pages is None:
            pages = [None] * len(text_chunks)
        if len(pages) != len(text_chunks):
            raise ValueError(
                f"{len(text_chunks)} chunks but {len(pages)} pages; a citation would be wrong"
            )
        self.pages.extend(pages)

        print(f"Session ingested {self.index.ntotal} chunks.")

    async def query(self, query_text: str, k: int = 8) -> list[dict]:
        """
        Performs a similarity search against the session's document chunks.

        Args:
            query_text: The user's question.
            k: The number of top results to retrieve.

        Returns:
            A list of dictionaries, each containing the 'text' of a relevant
            chunk and its similarity 'score'.
        """
        import asyncio
        if self.index.ntotal == 0:
            return []

        # Embed the query.
        try:
            query_embedding_raw = await asyncio.wait_for(
                asyncio.to_thread(self.embedding_model.encode, [query_text], convert_to_numpy=True),
                timeout=30.0
            )
            query_embedding = np.ascontiguousarray(query_embedding_raw.astype('float32'))
            faiss.normalize_L2(query_embedding)
        except asyncio.TimeoutError:
            raise TimeoutError("Embedding generation for query timed out.")

        # Search the index. With a normalised inner-product index, `scores` are
        # cosine similarities in [-1, 1], already sorted high to low.
        try:
            similarities, indices = await asyncio.wait_for(
                asyncio.to_thread(self.index.search, query_embedding, min(k, self.index.ntotal)),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise TimeoutError("FAISS search for query timed out.")

        results = []
        for i, vector_id in enumerate(indices[0]):
            if vector_id != -1:
                # Already a cosine similarity. Reported as-is rather than
                # rescaled, because the last thing this number did was get
                # dressed up as a confidence percentage it had no right to.
                score = similarities[0][i]

                results.append({
                    "text": self.chunks[vector_id],
                    "score": float(score),
                    "page": self.pages[vector_id] if vector_id < len(self.pages) else None,
                })

        return results

    def touch(self):
        """Updates the last_accessed timestamp to the current time."""
        self.last_accessed = datetime.datetime.now()
