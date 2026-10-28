import uuid
from datetime import datetime
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from memoryos.config import MemoryConfig, ModelConfig, StorageConfig


@dataclass
class MemoryEntry:
    """A single entry stored in long-term memory."""
    id: str
    content: str
    role: str
    turn_index: int
    timestamp: str
    relevance_score: float = 0.0  # populated on retrieval


class LongTermMemory:
    """
    Persistent semantic memory backed by ChromaDB.

    Stores conversation turns as embeddings.
    Retrieves relevant memories by semantic similarity.
    Persists to disk — survives process restarts.
    """

    def __init__(
        self,
        memory_config: MemoryConfig,
        model_config: ModelConfig,
        storage_config: StorageConfig,
        session_id: str,
    ) -> None:
        self.top_k = memory_config.long_term_top_k
        self.similarity_threshold = memory_config.long_term_similarity_threshold
        self.embedding_model = model_config.embedding_model
        self.session_id = session_id

        # OpenAI client for embeddings
        self._client = OpenAI()

        # ChromaDB client — persists to disk
        self._chroma = chromadb.PersistentClient(
            path=storage_config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Each session gets its own collection namespace
        collection_name = f"{storage_config.collection_name}_{session_id}"
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )

    def _embed(self, text: str) -> list[float]:
        """Convert text to embedding vector via OpenAI API."""
        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def add(self, role: str, content: str, turn_index: int) -> str:
        """
        Embed and store a message in ChromaDB.
        Returns the memory ID.
        """
        memory_id = str(uuid.uuid4())
        embedding = self._embed(content)

        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "role": role,
                "turn_index": turn_index,
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": self.session_id,
            }]
        )
        return memory_id

    def retrieve(self, query: str) -> list[MemoryEntry]:
        """
        Find the most semantically relevant memories for a query.
        Filters by similarity threshold before returning.
        """
        if self._collection.count() == 0:
            return []

        query_embedding = self._embed(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        memories = []
        for i, doc in enumerate(results["documents"][0]):
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score (1=identical, 0=unrelated)
            distance = results["distances"][0][i]
            similarity = 1 - (distance / 2)

            if similarity >= self.similarity_threshold:
                meta = results["metadatas"][0][i]
                memories.append(MemoryEntry(
                    id=results["ids"][0][i],
                    content=doc,
                    role=meta["role"],
                    turn_index=meta["turn_index"],
                    timestamp=meta["timestamp"],
                    relevance_score=round(similarity, 3),
                ))

        # Return sorted by relevance, most relevant first
        return sorted(memories, key=lambda m: m.relevance_score, reverse=True)

    @staticmethod
    def format_memories(memories: list[MemoryEntry]) -> str:
        """
        Format an already-retrieved list of memories into a context block.

        Pure function — no embedding, no I/O. Split out from to_prompt_format()
        so a caller (the coordinator) can call retrieve() ONCE, keep the
        structured list with scores for the trace, and still format the same
        block without a second embedding call.
        """
        if not memories:
            return ""

        lines = ["Relevant context from past conversations:"]
        for mem in memories:
            lines.append(
                f"- [{mem.role}] (relevance: {mem.relevance_score}): {mem.content}"
            )
        return "\n".join(lines)

    def to_prompt_format(self, query: str) -> str:
        """
        Retrieve relevant memories and format them as a
        context block for injection into the system prompt.

        Behaviour unchanged — now just composes retrieve() + format_memories().
        """
        return self.format_memories(self.retrieve(query))

    def count(self) -> int:
        return self._collection.count()

    def __repr__(self) -> str:
        return (
            f"LongTermMemory("
            f"session={self.session_id}, "
            f"entries={self.count()})"
        )