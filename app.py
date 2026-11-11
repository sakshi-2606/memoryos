import streamlit as st
from memoryos.agent import MemoryAgent
from memoryos.config import AppConfig, MemoryConfig

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MemoryOS",
    page_icon="🧠",
    layout="wide",
)

# ── Session state init ─────────────────────────────────────────────────────

def init_session():
    if "agent" not in st.session_state:
        config = AppConfig(
            memory=MemoryConfig(
                short_term_token_budget=600,   # small demo budget → eviction + episodic summaries show up within a few turns
                long_term_top_k=3,
            )
        )
        st.session_state.agent = MemoryAgent(config, session_id="streamlit-demo")
        st.session_state.messages = []
        st.session_state.memory_log = []

init_session()

# ── Layout ─────────────────────────────────────────────────────────────────

st.title("🧠 MemoryOS")
st.caption("Tiered agent memory — short-term · long-term · episodic")

col_chat, col_memory = st.columns([3, 2])

# ── Chat column ────────────────────────────────────────────────────────────

with col_chat:
    st.subheader("Conversation")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Say something...")

    if user_input:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.agent.chat(user_input)

            st.write(result["response"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"]
            })

        # Log memory metadata for visualiser
        st.session_state.memory_log.append({
            "turn": result["turn"],
            "message": user_input[:50] + "..." if len(user_input) > 50 else user_input,
            "metadata": result["memory"],
        })

        st.rerun()

# ── Memory visualiser column ───────────────────────────────────────────────

with col_memory:
    st.subheader("Memory State")

    agent = st.session_state.agent
    memory_state = agent.get_memory_state()

    # Overview metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Short-term", f"{len(agent.memory.short_term)} turns")
    m2.metric("Long-term", f"{agent.memory.long_term.count()} entries")
    m3.metric("Episodes", f"{len(agent.memory.episodic)}")

    st.divider()

    # Short-term buffer
    with st.expander("Short-term buffer", expanded=True):
        short_term = memory_state["short_term"]
        if short_term:
            for msg in short_term:
                role_icon = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(
                    f"`turn {msg['turn']}` {role_icon} {msg['content'][:80]}"
                    + ("..." if len(msg['content']) > 80 else "")
                )
        else:
            st.caption("Empty")

    # Episodes
    with st.expander("Episodic memory", expanded=True):
        episodes = memory_state["episodes"]
        if episodes:
            for ep in episodes:
                st.markdown(f"**{ep['id']}** (turns {ep['turns'][0]}–{ep['turns'][1]})")
                st.caption(ep["summary"])
        else:
            st.caption("No episodes yet")

    # Per-turn memory activity log
    with st.expander("Memory activity log", expanded=True):
        if st.session_state.memory_log:
            for entry in reversed(st.session_state.memory_log):
                meta = entry["metadata"]
                turn = entry["turn"]
                stored_lt = "✓" if meta["added_to_long_term"] else "✗"
                episode = "✓" if meta["episode_created"] else "✗"
                tiers = []
                if meta["context_used"]["episodic"]:
                    tiers.append("episodic")
                if meta["context_used"]["long_term"]:
                    tiers.append("long-term")
                if meta["context_used"]["short_term"]:
                    tiers.append("short-term")

                st.markdown(
                    f"`Turn {turn}` **{entry['message']}**\n\n"
                    f"Stored in LT: {stored_lt} · "
                    f"Episode: {episode} · "
                    f"Tiers used: {', '.join(tiers) if tiers else 'none'}"
                )
                st.divider()
        else:
            st.caption("No activity yet — send a message")

    # Reset button
    st.divider()
    if st.button("Reset session", type="secondary"):
        for key in ["agent", "messages", "memory_log"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()