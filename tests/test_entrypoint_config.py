"""
API-free test: entrypoint configs are built from tested factories, so a config
schema change breaks a test here instead of a user's `streamlit run`.

This is the guard for the drift class that bit twice — app.py building MemoryConfig
with removed fields, and a dropped ObservabilityConfig that agent.py still read.

Run from repo root (venv active):
    python -m pytest tests/test_entrypoint_config.py -v
"""

from memoryos.config import AppConfig, demo_config


def test_demo_config_constructs():
    # app.py runs with this exact config; constructing it here fails CI if the
    # MemoryConfig schema drifts out from under the Streamlit entrypoint.
    cfg = demo_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.memory.short_term_token_budget == 600
    assert cfg.memory.long_term_top_k == 3


def test_appconfig_exposes_every_section_entrypoints_read():
    # agent.py reads config.observability / config.trace; api.py reads config.model
    # etc. Assert the full section contract so a removed section is caught.
    cfg = AppConfig()
    for section in ("memory", "model", "storage", "trace", "observability"):
        assert getattr(cfg, section, None) is not None, f"AppConfig missing '{section}'"