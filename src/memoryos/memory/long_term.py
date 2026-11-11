import uuid
from datetime import datetime
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from memoryos.config import MemoryConfig, ModelConfig, StorageConfig
from memoryos.memory.forgetting import ForgettingManager, EvictionCandidate
from memoryos.session_id import validate_session_id, collection_name_for


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

    Stores conversation turns as embeddings, retrieves by semantic similarity,
    and — when the store exceeds capacity — forgets the lowest-retention memories
    via the Hybrid policy ported from MemoryBench (its proven-best forgetting
    strategy). Persists to disk; survives process restarts.
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
        # validate at the store boundary too (defence in depth beyond the API edge)
        self.session_id = validate_session_id(session_id)

        # Forgetting policy (MemoryBench's winner by default: Hybrid)
        self._forgetting = ForgettingManager(
            policy=memory_config.forgetting_policy,
            max_episodes=memory_config.long_term_max_episodes,
            hybrid_weights=memory_config.forgetting_hybrid_weights,
        )

        self._client = OpenAI()
        self._chroma = chromadb.PersistentClient(
            path=storage_config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        collection_name = collection_name_for(storage_config.collection_name, self.session_id)
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, text: str) -> list[float]:
        """Convert text to embedding vector via OpenAI API."""
        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def add(self, role: str, content: str, turn_index: int, importance: float = 1.0) -> str:
        """
        Embed and store a message in ChromaDB. Returns the memory ID.

        `importance` (0.0–2.0, default neutral 1.0) feeds the Hybrid forgetting
        policy; it can be reweighted later via update_importance().
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
                "importance": importance,
            }],
        )
        return memory_id

    def retrieve(self, query: str) -> list[MemoryEntry]:
        """Find the most semantically relevant memories, filtered by threshold."""
        if self._collection.count() == 0:
            return []

        query_embedding = self._embed(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for i, doc in enumerate(results["documents"][0]):
            # ChromaDB cosine distance = 1 - cosine_similarity, so recover true
            # cosine directly (1=identical, 0=unrelated).
            distance = results["distances"][0][i]
            similarity = 1 - distance

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

        return sorted(memories, key=lambda m: m.relevance_score, reverse=True)

    @staticmethod
    def format_memories(memories: list[MemoryEntry]) -> str:
        """Format an already-retrieved list of memories into a context block."""
        if not memories:
            return ""
        lines = ["Relevant context from past conversations:"]
        for mem in memories:
            lines.append(
                f"- [{mem.role}] (relevance: {mem.relevance_score}): {mem.content}"
            )
        return "\n".join(lines)

    def to_prompt_format(self, query: str) -> str:
        """Retrieve relevant memories and format them (retrieve + format_memories)."""
        return self.format_memories(self.retrieve(query))

    # ── forgetting ────────────────────────────────────────────────────────────

    def _all_candidates(self) -> list[EvictionCandidate]:
        """Every stored memory as an EvictionCandidate (for the forgetting policy)."""
        if self._collection.count() == 0:
            return []
        res = self._collection.get(include=["documents", "metadatas"])
        candidates = []
        for i, doc in enumerate(res["documents"]):
            meta = res["metadatas"][i]
            candidates.append(EvictionCandidate(
                episode_id=res["ids"][i],
                content=doc,
                timestamp=float(meta.get("turn_index", 0)),   # monotonic recency proxy
                importance=float(meta.get("importance", 1.0)),
            ))
        return candidates

    def maybe_forget(self) -> list[EvictionCandidate]:
        """
        If the store is over capacity, evict the lowest-retention memories per the
        configured policy. Returns the evicted candidates (for the trace).
        """
        candidates = self._all_candidates()
        evict_ids = self._forgetting.candidates_to_evict(candidates)
        if not evict_ids:
            return []
        self._collection.delete(ids=evict_ids)
        evict_set = set(evict_ids)
        return [c for c in candidates if c.episode_id in evict_set]

    def update_importance(self, memory_id: str, importance: float) -> None:
        """Reweight a stored memory's importance (e.g. a user flag)."""
        self._collection.update(ids=[memory_id], metadatas=[{"importance": importance}])

    def count(self) -> int:
        return self._collection.count()

    def __repr__(self) -> str:
        return (
            f"LongTermMemory("
            f"session={self.session_id}, "
            f"entries={self.count()}, "
            f"policy={self._forgetting.policy.value})"
        )