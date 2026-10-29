"""
Threshold calibration sweep for long-term retrieval (F2).

Measures TRUE cosine similarity (1 - chroma_distance) for every probe x fact pair
in the eval dataset, then sweeps candidate thresholds and reports precision/recall
across all probes. Recommends the operating point:

    the highest threshold at which recall is still 1.0 for EVERY probe
    (= max precision without ever dropping a relevant memory)

This is a measurement tool — it does not modify any system code. It writes a
reproducible results file next to itself.

Run from repo root (venv active):
    python scripts/threshold_sweep.py
"""

import uuid
from collections import defaultdict
from pathlib import Path

from memoryos.config import AppConfig
from memoryos.memory.long_term import LongTermMemory
from memoryos.memeval import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATASET = load_dataset(ROOT / "tests" / "memeval_dataset.json")
RESULTS_PATH = Path(__file__).resolve().parent / "threshold_sweep_results.md"

THRESHOLDS = [round(0.05 * i, 2) for i in range(0, 19)]   # 0.00 .. 0.90


def _plant(session: str, facts) -> LongTermMemory:
    cfg = AppConfig()
    lt = LongTermMemory(
        memory_config=cfg.memory,
        model_config=cfg.model,
        storage_config=cfg.storage,
        session_id=f"sweep-{session}-{uuid.uuid4().hex[:8]}",
    )
    for f in facts:
        lt.add("user", f.content, turn_index=0)
    return lt


def _true_similarities(lt: LongTermMemory, query: str, facts) -> dict[str, float]:
    """Return {fact_id: true_cosine_similarity} for one query against all facts."""
    emb = lt._embed(query)
    res = lt._collection.query(
        query_embeddings=[emb],
        n_results=lt._collection.count(),
        include=["documents", "distances"],
    )
    by_content = {f.content: f.id for f in facts}
    sims: dict[str, float] = {}
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
        if doc in by_content:
            sims[by_content[doc]] = 1.0 - dist      # true cosine
    return sims


def main() -> None:
    # 1. gather true-cosine similarities per probe
    by_session = defaultdict(list)
    for p in DATASET.retrieval_probes:
        by_session[p.session].append(p)

    planted: list[LongTermMemory] = []
    per_probe = []   # (probe, {fact_id: sim}, relevant_set)
    try:
        for session, probes in by_session.items():
            facts = DATASET.facts_for_session(session)
            lt = _plant(session, facts)
            planted.append(lt)
            for p in probes:
                sims = _true_similarities(lt, p.query, facts)
                per_probe.append((p, sims, set(p.relevant)))

        # 2. per-probe separation (useful context)
        sep_lines = ["", "Per-probe separation (true cosine):", ""]
        for p, sims, relevant in per_probe:
            rel_sims = [sims[r] for r in relevant if r in sims]
            dis_sims = [s for fid, s in sims.items() if fid not in relevant]
            sep_lines.append(
                f"  {p.id}: relevant={min(rel_sims):.3f} | "
                f"top distractor={max(dis_sims):.3f} | "
                f"margin={min(rel_sims) - max(dis_sims):+.3f}"
            )

        # 3. sweep
        header = f"{'thresh':>7} | {'avg_prec':>8} | {'avg_rec':>7} | recall=1 all?"
        sep = "-" * len(header)
        rows = [header, sep]
        candidates = []
        for t in THRESHOLDS:
            precs, recs, all_rec_1 = [], [], True
            for p, sims, relevant in per_probe:
                retrieved = {fid for fid, s in sims.items() if s >= t}
                tp = retrieved & relevant
                rec = len(tp) / len(relevant)
                prec = (len(tp) / len(retrieved)) if retrieved else 0.0
                recs.append(rec)
                precs.append(prec)
                if rec < 1.0:
                    all_rec_1 = False
            avg_p = sum(precs) / len(precs)
            avg_r = sum(recs) / len(recs)
            rows.append(
                f"{t:>7.2f} | {avg_p:>8.3f} | {avg_r:>7.3f} | {'yes' if all_rec_1 else 'no'}"
            )
            if all_rec_1:
                candidates.append((t, avg_p))

        # 4. operating point: highest threshold with full recall (max precision)
        recommended = max(candidates, key=lambda c: c[0]) if candidates else (0.0, 0.0)

        # 5. report
        out = []
        out.append("# Threshold calibration sweep (F2)")
        out.append("")
        out.append(f"Dataset: v{DATASET.dataset_version} — "
                   f"{len(DATASET.retrieval_probes)} probes, "
                   f"{len(DATASET.facts)} facts")
        out.append("Similarity: true cosine = 1 - chroma_cosine_distance")
        out.append("Operating-point rule: highest threshold with recall=1.0 for every probe")
        out.append("")
        out.append("```")
        out.extend(rows)
        out.extend(sep_lines)
        out.append("")
        out.append(f"RECOMMENDED THRESHOLD: {recommended[0]:.2f} "
                   f"(avg precision {recommended[1]:.3f}, recall 1.0)")
        out.append("```")
        report = "\n".join(out)

        print(report)
        RESULTS_PATH.write_text(report + "\n", encoding="utf-8")
        print(f"\n(written to {RESULTS_PATH})")

    finally:
        for lt in planted:
            try:
                lt._chroma.delete_collection(lt._collection.name)
            except Exception:
                pass


if __name__ == "__main__":
    main()