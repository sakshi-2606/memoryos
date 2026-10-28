"""
API-free test: the eval dataset loads and passes schema + referential validation.

No OpenAI, no chromadb — just JSON loading and integrity checks. This guards the
fixture the four probe-suites depend on.

Run from repo root (venv active):
    python tests/test_memeval_dataset.py
"""

from pathlib import Path

from memoryos.memeval import load_dataset, DATASET_VERSION

DATASET_PATH = Path(__file__).parent / "memeval_dataset.json"


def main() -> None:
    ds = load_dataset(DATASET_PATH)

    assert ds.dataset_version == DATASET_VERSION
    assert len(ds.facts) >= 5
    assert ds.retrieval_probes, "no retrieval probes"
    assert ds.leakage_probes, "no leakage probes"
    assert ds.episodic_cases, "no episodic cases"

    # referential integrity (load_dataset already asserts this; re-affirm a slice)
    ids = {f.id for f in ds.facts}
    for p in ds.retrieval_probes:
        assert set(p.relevant) <= ids
    for lk in ds.leakage_probes:
        assert lk.fact in ids
        assert lk.planted_session != lk.query_session

    # convenience lookups work
    assert ds.fact_by_id("f1").content
    assert len(ds.facts_for_session("s3")) == 3

    print(f"✓ dataset v{ds.dataset_version} loaded and valid")
    print(f"  {len(ds.facts)} facts | "
          f"{len(ds.retrieval_probes)} retrieval | "
          f"{len(ds.leakage_probes)} leakage | "
          f"{len(ds.episodic_cases)} episodic case(s)")


if __name__ == "__main__":
    main()