import datetime
from typing import Dict
from rag_session import RAGSession

class UserSession:
    """
    Manages a collection of documents for a single user.

    This class holds multiple RAGSession objects, each corresponding to an
    ingested document. It also tracks the last access time for the entire
    user session.
    """
    def __init__(self):
        self.last_accessed: datetime.datetime = datetime.datetime.now()
        self.docs: Dict[str, RAGSession] = {}

    def add_doc(self, doc_id: str, rag_session: RAGSession):
        """Adds a new document session to this user's collection."""
        self.docs[doc_id] = rag_session
        self.touch()

    def remove_doc(self, doc_id: str):
        """Removes a document session from this user's collection."""
        if doc_id in self.docs:
            del self.docs[doc_id]
            self.touch()

    def get_doc(self, doc_id: str) -> RAGSession | None:
        """Retrieves a specific document session."""
        return self.docs.get(doc_id)

    def get_all_docs(self) -> list[RAGSession]:
        """Returns a list of all document sessions for this user."""
        return list(self.docs.values())

    def touch(self):
        """Updates the last_accessed timestamp to the current time."""
        self.last_accessed = datetime.datetime.now()
