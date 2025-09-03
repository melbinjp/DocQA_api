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

        # Create an in-memory FAISS index. IndexFlatL2 is a simple, fast index
        # for dense vectors.
        self.index = faiss.IndexFlatL2(d_model)

        # In-memory store for the actual text chunks corresponding to the vectors.
        # The index in this list is the ID used in the FAISS index.
        self.chunks = []

    def ingest(self, text_chunks: list[str], embeddings: np.ndarray):
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

        # FAISS requires a flat numpy array of float32.
        embeddings_float32 = np.array(embeddings, dtype='float32')

        # Add the new embeddings to the FAISS index.
        self.index.add(embeddings_float32)

        # Store the corresponding text chunks.
        self.chunks.extend(text_chunks)

        print(f"Session ingested {self.index.ntotal} chunks.")

    def query(self, query_text: str, k: int = 5) -> list[dict]:
        """
        Performs a similarity search against the session's document chunks.

        Args:
            query_text: The user's question.
            k: The number of top results to retrieve.

        Returns:
            A list of dictionaries, each containing the 'text' of a relevant
            chunk and its similarity 'score'.
        """
        if self.index.ntotal == 0:
            return []

        # Embed the query.
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True).astype('float32')

        # Search the index. `distances` are L2 distances, `indices` are the
        # integer IDs of the vectors in the index.
        distances, indices = self.index.search(query_embedding, k=min(k, self.index.ntotal))

        results = []
        for i, vector_id in enumerate(indices[0]):
            if vector_id != -1:
                # A simple conversion from L2 distance to a normalized similarity score (0-1).
                # This is a basic heuristic.
                score = 1 / (1 + distances[0][i])

                results.append({
                    "text": self.chunks[vector_id],
                    "score": float(score)
                })

        return results

    def touch(self):
        """Updates the last_accessed timestamp to the current time."""
        self.last_accessed = datetime.datetime.now()
