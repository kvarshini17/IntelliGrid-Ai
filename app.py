"""
app.py
--------
Streamlit entry point for IntelliGrid AI (Phase 9).

Loads the processed dataset once, runs the full deterministic pipeline once
(cached), and routes to the five pages. Contains minimal logic of its own --
delegates everything to modules/ and views/.

Run with:
    streamlit run app.py
"""

import logging
import traceback
from pathlib import Path

import streamlit as st

import config
from llm.provider import get_provider
from modules.data_manager import DataLoadError, load_processed_dataset
from modules.pipeline import pipeline_to_summary_dataframe, run_full_pipeline
from views import ai_assistant, about, dashboard, energy_analysis, optimization

st.set_page_config(page_title="IntelliGrid AI", page_icon="⚡", layout="wide")

logger = logging.getLogger(__name__)

PAGES = {
    "Dashboard": dashboard,
    "Energy Analysis": energy_analysis,
    "Optimization": optimization,
    "AI Assistant": ai_assistant,
}


def _render_page(page: str, df, summary_df, pipeline_results) -> None:
    """
    Render one page inside an error boundary. A failure shows an actionable
    message (and a traceback in an expander) instead of a blank content area.
    Streamlit's own control-flow exceptions (st.rerun / st.stop) derive from
    BaseException, so they pass through this `except Exception` untouched.
    """
    try:
        if page == "About":
            about.render()
        else:
            PAGES[page].render(df, summary_df, pipeline_results)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Page %r failed to render", page)
        st.error(
            f"The **{page}** page failed to render: `{type(exc).__name__}: {exc}`. "
            "Other pages are unaffected. Details are below and in the terminal log; "
            "if this persists, re-run `pytest tests/test_app.py` to locate the failing component."
        )
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


@st.cache_data(show_spinner="Loading dataset...")
def _load_data():
    return load_processed_dataset()


@st.cache_data(show_spinner="Running condition / wastage / anomaly / optimization / recommendation pipeline...")
def _run_pipeline(df):
    results = run_full_pipeline(df)
    summary = pipeline_to_summary_dataframe(results)
    return results, summary


def main():
    st.title("⚡ IntelliGrid AI")
    st.caption("Explainable Energy Consumption Optimization for Smart Buildings")

    # A pages/ folder next to app.py makes Streamlit add every file in it to the sidebar as its
    # own page. The files in views/ only define render(), so those entries would be blank.
    if (Path(__file__).resolve().parent / "pages").exists():
        st.sidebar.warning(
            "A `pages/` folder exists next to app.py (probably left over from an older copy of the "
            "project). Streamlit lists its files as extra sidebar pages, and they render blank. "
            "Delete the `pages/` folder -- the real pages live in `views/` and are chosen with the "
            "Navigation radio below."
        )

    try:
        df = _load_data()
    except DataLoadError as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    try:
        pipeline_results, summary_df = _run_pipeline(df)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis pipeline failed")
        st.error(f"The analysis pipeline failed, so no page can be shown: `{type(exc).__name__}: {exc}`.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        st.stop()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Energy Analysis", "Optimization", "AI Assistant", "About"])

    provider = get_provider()
    ready, detail = provider.status()
    if provider.name == "null":
        st.sidebar.info("No LLM configured (default). The AI Assistant shows retrieved knowledge-base "
                        "passages, not generated text.")
    elif not ready:
        st.sidebar.warning(f"LLM provider: {provider.name} is selected but not ready ({detail}) "
                           "The AI Assistant shows retrieved passages, labelled as not LLM-generated.")
    else:
        st.sidebar.info(f"LLM provider: {provider.name}. If it is unreachable, the AI Assistant "
                        "shows retrieved passages instead, labelled as not LLM-generated.")

    _render_page(page, df, summary_df, pipeline_results)


if __name__ == "__main__":
    main()
