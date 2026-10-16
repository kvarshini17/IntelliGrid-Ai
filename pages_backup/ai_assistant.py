"""
pages/ai_assistant.py
------------------------
Purpose : AI Assistant page -- chat interface combining structured project
          results (optionally for a selected record) with RAG-retrieved
          knowledge and an LLM (or Null-provider) explanation. Sources are
          always shown when knowledge-base information is used.
"""

import streamlit as st

from rag.rag_pipeline import answer_with_rag


def _build_system_context(record: dict) -> str:
    return (
        f"Record {record['record_id']} -- Room {record['room_id']}, {record['timestamp']}.\n"
        f"Condition: {record['condition_label']}.\n"
        f"Potential wastage: {record['potential_wastage']} ({record['wastage_type']}, "
        f"severity={record['severity']}). Reason: {record['reason']}\n"
        f"Anomaly: {record['anomaly_type']}. Reason: {record['anomaly_reason']}\n"
        f"Energy consumption: {record['energy_consumption']:.2f}, "
        f"Tariff: {record['electricity_tariff']} ({record['tariff_category']})."
    )


def render(df, summary_df, pipeline_results: list):
    st.header("AI Assistant")
    st.caption(
        "Ask energy-efficiency questions, optionally grounded in a specific record's results. "
        "Answers are retrieved from the project's knowledge base; an LLM (if configured) phrases "
        "the answer using only that retrieved context. Deterministic analysis results are never "
        "overridden by the assistant."
    )

    use_record = st.checkbox("Ground my question in a specific record")
    selected_record = None
    if use_record:
        record_ids = summary_df["record_id"].tolist()
        selected_id = st.selectbox("Select a record", record_ids)
        selected_record = next(r for r in pipeline_results if r["record_id"] == selected_id)
        with st.expander("Record context used", expanded=False):
            st.text(_build_system_context(selected_record))

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            st.caption(f"mode: {entry['mode']}")
            if entry.get("sources"):
                with st.expander("Knowledge sources used"):
                    for src in entry["sources"]:
                        st.markdown(f"**{src['title']}** (relevance score: {src['score']:.2f})")
                        st.caption(src["text"])
            elif entry["mode"] == "no_evidence":
                st.info("No sufficiently relevant knowledge-base evidence was found for this question.")

    question = st.chat_input("Ask a question, e.g. 'Why should lighting be turned off in unoccupied rooms?'")
    if question:
        system_context = _build_system_context(selected_record) if selected_record else None
        with st.spinner("Retrieving knowledge and generating an answer..."):
            result = answer_with_rag(question, system_context=system_context)
        st.session_state.chat_history.append({
            "question": question, "answer": result["answer"],
            "mode": result["mode"], "sources": result.get("sources", []),
        })
        st.rerun()

    st.divider()
    st.markdown("**Example questions:**")
    st.markdown("- Why should HVAC be reduced when a room is unoccupied?")
    st.markdown("- Why should lighting be turned off in unoccupied rooms?")
    st.markdown("- What energy-efficiency practices are relevant here?")
    st.markdown("- How can building energy consumption be reduced?")
