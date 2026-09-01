import datetime
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# How many candidates each retriever contributes before fusion. Wider than the
# k finally returned, because the point of fusion is to let a chunk that one
# retriever ranked eighth and the other ranked second come out near the top.
CANDIDATE_POOL = 30

# The dense side searches vectors, and a long chunk owns several of them, so a
# pool of 30 vectors can collapse to six or seven distinct chunks. This is the
# vector pool, sized so the number of distinct chunks reaching fusion stays in
# the same range as the lexical side's.
DENSE_VECTOR_POOL = 120

# The constant in reciprocal rank fusion, from Cormack et al. 60 is the value
# the paper uses and the one every implementation inherits; it damps the
# difference between rank 1 and rank 2 so a single confident retriever cannot
# monopolise the result.
RRF_K = 60

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Words and numbers, lowercased. Numbers matter here more than usual: the
    facts that dense retrieval misses in this corpus are table cells."""
    return _TOKEN.findall(text.lower())

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

        # In-memory store for the actual text chunks. This is NO LONGER parallel
        # to the FAISS index: one chunk can own several vectors, so a vector id
        # is mapped through self.vector_parent to get here.
        self.chunks = []

        # vector id -> index into self.chunks.
        #
        # A long chunk is indexed once whole and once per window across it, so a
        # fact buried inside a chunk about something else is still reachable. The
        # RAG paper's "21M documents" sat in a 1435-character chunk that opens
        # "To estimate the probability of an hypothesis y"; three differently
        # worded questions about index size all failed to retrieve it, because
        # the chunk's vector is about marginalising over documents. Whichever
        # vector matches, the parent chunk is what is returned, read and cited.
        self.vector_parent = []

        # parent index -> one of its vector ids, so a chunk found only by the
        # lexical retriever can still have its cosine recovered for display.
        self.parent_vector = {}

        # Lexical half of the retriever, rebuilt whenever chunks are added.
        #
        # Dense embeddings are the wrong tool for part of a document and it is a
        # measurable part. Asked "how does the parameter count of the big model
        # compare to the base model", where the answer is a column headed
        # `params x10^6` holding 65 and 213, the dense retriever pulled the prose
        # of that page and never the table, and the API answered that it had no
        # such information twice: once with 500-character chunks and again with
        # 1500-character chunks and the table rendered as a grid. The chunk
        # existed and was correct both times. A block of bare numbers simply has
        # no embedding near "parameter count".
        #
        # `params` is right there as a literal token, which is what BM25 is for.
        self.bm25 = None
        self.tokenized = []

        # Parallel to self.chunks: the page each chunk came from, or None for
        # formats without pages. Kept as a separate list rather than a dict so a
        # FAISS vector id indexes both without a lookup that could disagree.
        self.pages = []

    def ingest(self, text_chunks: list[str], embeddings: np.ndarray,
               pages: list | None = None, vector_parents: list | None = None):
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

        # One vector per row of `embeddings`, each owned by a chunk. Without an
        # explicit mapping the relationship is one to one, which is what a caller
        # that knows nothing about windows should get.
        if vector_parents is None:
            vector_parents = list(range(len(text_chunks)))
        if len(vector_parents) != len(embeddings_float32):
            raise ValueError(
                f"{len(embeddings_float32)} vectors but {len(vector_parents)} owners; "
                "a citation would name the wrong chunk"
            )

        base_vector = self.index.ntotal
        base_chunk = len(self.chunks)

        # Add the new embeddings to the FAISS index.
        self.index.add(embeddings_float32)

        for offset, parent in enumerate(vector_parents):
            absolute_parent = base_chunk + parent
            self.vector_parent.append(absolute_parent)
            self.parent_vector.setdefault(absolute_parent, base_vector + offset)

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

        # Rebuilt rather than updated: BM25 scoring depends on corpus-wide
        # document frequencies and average length, so an index built over the
        # first document would score the second against the wrong statistics.
        self.tokenized.extend(_tokenize(c) for c in text_chunks)
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

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
        pool = min(CANDIDATE_POOL, len(self.chunks))
        vector_pool = min(DENSE_VECTOR_POOL, self.index.ntotal)
        try:
            similarities, indices = await asyncio.wait_for(
                asyncio.to_thread(self.index.search, query_embedding, vector_pool),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise TimeoutError("FAISS search for query timed out.")

        # Collapse vectors to the chunks that own them, keeping the best rank
        # and the best cosine each chunk achieved. Several windows of one chunk
        # can all match; that is one result, not four.
        dense_ranked, cosine = [], {}
        for position, vector_id in enumerate(indices[0]):
            if vector_id == -1:
                continue
            parent = self.vector_parent[int(vector_id)]
            similarity = float(similarities[0][position])
            if parent not in cosine or similarity > cosine[parent]:
                cosine[parent] = similarity
            if parent not in dense_ranked:
                dense_ranked.append(parent)

        # Lexical ranking, which is over whole chunks already.
        lexical_ranked = []
        if self.bm25 is not None:
            bm25_scores = await asyncio.to_thread(self.bm25.get_scores, _tokenize(query_text))
            lexical_ranked = [int(i) for i in np.argsort(bm25_scores)[::-1][:pool]
                              if bm25_scores[i] > 0]

        # Reciprocal rank fusion. Ranks rather than scores, because a cosine and
        # a BM25 score are not on the same scale and never will be; normalising
        # them against each other would be inventing a relationship.
        fused: dict[int, float] = {}
        for ranking in (dense_ranked, lexical_ranked):
            for rank, parent in enumerate(ranking):
                fused[parent] = fused.get(parent, 0.0) + 1.0 / (RRF_K + rank + 1)

        order = sorted(fused, key=lambda p: fused[p], reverse=True)[:k]

        results = []
        for parent in order:
            # The reported score stays a cosine, so it means one thing wherever
            # it appears. A chunk only the lexical side found has no cosine yet,
            # so it is recovered from one of that chunk's stored vectors rather
            # than left blank or filled with a fused rank score that would look
            # like a similarity and not be one.
            score = cosine.get(parent)
            if score is None:
                try:
                    vec = self.index.reconstruct(self.parent_vector[parent])
                    score = float(np.dot(query_embedding[0], vec))
                except Exception:
                    score = 0.0

            results.append({
                "text": self.chunks[parent],
                "score": float(score),
                "page": self.pages[parent] if parent < len(self.pages) else None,
            })

        return results

    def touch(self):
        """Updates the last_accessed timestamp to the current time."""
        self.last_accessed = datetime.datetime.now()
