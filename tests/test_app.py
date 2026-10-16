"""
tests/test_app.py
---------------------
Streamlit integration tests (Phase 9/10) using streamlit.testing.v1.AppTest.
Confirms the application actually runs end-to-end for every page, not just
that its source compiles.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Dashboard", "Energy Analysis", "Optimization", "AI Assistant", "About"]


@pytest.mark.parametrize("page_name", PAGES)
def test_page_renders_without_exception(page_name):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page_name).run()
    assert len(at.exception) == 0, [str(e) for e in at.exception]


def test_dashboard_is_default_page_and_shows_metrics():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert len(at.exception) == 0
    assert len(at.metric) >= 5  # Total/Occupied/Unoccupied/Wastage/Anomalies KPIs


def test_sidebar_has_all_five_pages():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    options = list(at.sidebar.radio[0].options)
    assert set(PAGES).issubset(set(options))


def test_app_title_present():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    titles = [t.value for t in at.title]
    assert any("IntelliGrid" in t for t in titles)


# ---------------------------------------------------------------------------
# Regression tests for the blank "Optimization" page defect.
#
# Root cause: a folder named `pages/` next to app.py makes Streamlit register
# every file in it as an auto-generated navigation entry (legacy multipage
# mode, `_mpa_v1`). Those files only define `render()`, so opening one
# rendered nothing -- a blank page with no exception. The folder is now named
# `views/`. These tests guard against reintroducing that.
# ---------------------------------------------------------------------------
def test_no_streamlit_auto_discovered_pages_directory():
    assert not (Path(APP_PATH).parent / "pages").exists()


def test_streamlit_builds_no_automatic_navigation():
    import streamlit.commands.navigation as nav

    calls = []
    original = nav._navigation

    def spy(pages, **kwargs):
        calls.append([p.title for p in pages])
        return original(pages, **kwargs)

    nav._navigation = spy
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
    finally:
        nav._navigation = original
    assert calls == [], f"Streamlit auto-registered pages: {calls}"


@pytest.mark.parametrize("page_name", PAGES)
def test_every_page_renders_visible_content(page_name):
    """A page that raises no exception but renders nothing is still a defect."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page_name).run()
    assert len(at.exception) == 0
    assert len(at.header) >= 1


# ---------------------------------------------------------------------------
# Dashboard presentation: partial day, wastage/anomaly separation
# ---------------------------------------------------------------------------
import pandas as pd

import config
from views import ai_assistant as ai_view
from views import dashboard as dash


@pytest.fixture(scope="module")
def real_summary_df():
    from modules.data_manager import load_processed_dataset
    from modules.pipeline import pipeline_to_summary_dataframe, run_full_pipeline
    return pipeline_to_summary_dataframe(run_full_pipeline(load_processed_dataset()))


def test_daily_totals_flags_the_real_datasets_partial_last_day(real_summary_df):
    daily = dash._daily_totals(real_summary_df)
    partial = daily[daily["is_partial"]]
    assert len(daily) == 42
    assert len(partial) == 1
    assert partial.iloc[0]["date"] == pd.Timestamp("2022-02-11")
    assert partial.iloc[0]["records"] == 16
    assert (daily[~daily["is_partial"]]["records"] == 24).all()


def test_daily_totals_flags_any_day_with_fewer_than_24_records():
    ts = list(pd.date_range("2022-03-01 00:00", periods=24, freq="h")) + \
         list(pd.date_range("2022-03-02 00:00", periods=10, freq="h"))
    df = pd.DataFrame({"timestamp": ts, "energy_consumption": [1.0] * len(ts)})
    daily = dash._daily_totals(df)
    assert daily["is_partial"].tolist() == [False, True]
    assert daily["records"].tolist() == [24, 10]


def test_energy_chart_draws_partial_day_separately_and_labels_it(real_summary_df):
    fig = dash._energy_over_time_figure(dash._daily_totals(real_summary_df))
    names = [t.name for t in fig.data]
    assert any("Partial day" in n for n in names)
    full_trace = next(t for t in fig.data if "Full day" in t.name)
    assert len(full_trace.x) == 41  # the partial day is not part of the full-day line
    assert any("Partial day" in a.text for a in fig.layout.annotations)
    assert "dataset units" in fig.layout.yaxis.title.text


def test_wastage_and_anomaly_are_reported_as_independent_signals(real_summary_df):
    overlap = dash._signal_overlap_counts(real_summary_df)
    assert overlap == {"wastage_only": 70, "anomaly_only": 13, "both": 0}
    wastage = dash._wastage_table(real_summary_df)
    anomaly = dash._anomaly_table(real_summary_df)
    assert len(wastage) == 70 and len(anomaly) == 13
    # each table carries only its own signal's columns
    assert "z-score" not in wastage.columns
    assert "Wastage type" not in anomaly.columns


def test_dashboard_has_separate_wastage_and_anomaly_tabs():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    labels = [t.label for t in at.tabs]
    assert any(l.startswith("Potential wastage") for l in labels)
    assert any(l.startswith("Statistical anomalies") for l in labels)


# ---------------------------------------------------------------------------
# AI Assistant disclosure
# ---------------------------------------------------------------------------
def _assistant(monkeypatch, provider, question=None, record=False):
    monkeypatch.setattr(config, "LLM_PROVIDER", provider)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("AI Assistant").run()
    if record:
        at.checkbox[0].check().run()
    if question:
        at.chat_input[0].set_value(question).run()
    return at


def _all_text(at):
    parts = [e.value for e in at.info] + [e.value for e in at.caption] + \
            [e.value for e in at.warning] + [e.value for e in at.markdown]
    return "\n".join(parts)


def test_default_mode_states_no_llm_and_tfidf_fallback(monkeypatch):
    at = _assistant(monkeypatch, "null")
    text = _all_text(at)
    assert len(at.exception) == 0
    assert "No LLM is configured" in text
    assert "not LLM-generated" in text
    assert "TF-IDF" in text and "Sentence Transformers is not installed" in text


def test_default_mode_answer_is_labelled_not_llm_generated(monkeypatch):
    at = _assistant(monkeypatch, "null", "Why should lighting be turned off in unoccupied rooms?")
    text = _all_text(at)
    assert "Not LLM-generated" in text
    assert "Generated by the" not in text


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_unreachable_provider_falls_back_and_is_labelled_not_llm_generated(monkeypatch, provider):
    at = _assistant(monkeypatch, provider, "Why should lighting be turned off in unoccupied rooms?")
    text = _all_text(at)
    assert len(at.exception) == 0
    assert "The configured LLM was unavailable" in text
    assert "Not LLM-generated" in text
    assert "Generated by the" not in text


def test_selected_record_with_no_llm_warns_that_record_is_not_interpreted(monkeypatch):
    at = _assistant(monkeypatch, "null", record=True)
    assert any("does not interpret this record" in w.value for w in at.warning)


def test_answer_label_never_calls_retrieved_text_llm_generated():
    for mode in ("null", "fallback", "no_evidence"):
        assert "Generated by" not in ai_view._answer_label(mode)
    assert ai_view._answer_label("null").endswith("Not LLM-generated.")
    assert ai_view._answer_label("fallback").endswith("Not LLM-generated.")
    # only a real provider name produces the "Generated by" wording
    assert ai_view._answer_label("ollama").startswith("Generated by the 'ollama'")


def test_sources_are_shown_as_excerpts_not_complete_passages(monkeypatch):
    at = _assistant(monkeypatch, "null", "Why should lighting be turned off in unoccupied rooms?")
    excerpts = [c.value for c in at.caption if c.value.startswith("Excerpt:")]
    assert excerpts and all(e.endswith("...") for e in excerpts)


# ---------------------------------------------------------------------------
# Page-failure handling and Claude provider states in the UI
# ---------------------------------------------------------------------------
def test_page_failure_shows_actionable_error_not_a_blank_page(monkeypatch):
    from views import optimization as optimization_view

    def boom(*args, **kwargs):
        raise RuntimeError("simulated page failure")

    monkeypatch.setattr(optimization_view, "render", boom)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Optimization").run()
    assert len(at.exception) == 0  # handled, not an unhandled crash
    assert any("Optimization" in e.value and "failed to render" in e.value and "simulated page failure" in e.value
               for e in at.error)
    # the other pages are unaffected
    at.sidebar.radio[0].set_value("About").run()
    assert len(at.error) == 0 and len(at.header) >= 1


def test_leftover_pages_folder_triggers_a_visible_warning(monkeypatch):
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: True if self.name == "pages" else real_exists(self))
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert any("pages/" in w.value and "Delete" in w.value for w in at.sidebar.warning)


def test_claude_active_answer_is_labelled_as_generated_and_shows_sources(monkeypatch):
    from llm.provider import AnthropicProvider
    monkeypatch.setenv("ANTHROPIC_API_KEY", "FAKE-test-key-0123456789")
    monkeypatch.setattr(AnthropicProvider, "generate",
                        lambda self, prompt, system_prompt="": "Mocked Claude answer.")
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("AI Assistant").run()
    at.chat_input[0].set_value("Why should lighting be turned off in unoccupied rooms?").run()
    text = _all_text(at)
    assert len(at.exception) == 0
    assert any("LLM provider: anthropic" in i.value for i in at.sidebar.info)
    assert "Mocked Claude answer." in text
    assert "Generated by the 'anthropic' LLM provider" in text
    assert "Not LLM-generated" not in text
    assert [c for c in at.caption if c.value.startswith("Excerpt:")]


def test_claude_selected_without_key_warns_and_falls_back(monkeypatch):
    at = _assistant(monkeypatch, "anthropic", "Why should lighting be turned off in unoccupied rooms?")
    text = _all_text(at)
    assert len(at.exception) == 0
    assert any("not ready" in w.value and "ANTHROPIC_API_KEY" in w.value for w in at.warning)
    assert any("not ready" in w.value for w in at.sidebar.warning)
    assert "The configured LLM was unavailable" in text and "Not LLM-generated" in text
    assert "Generated by the" not in text


def test_no_key_default_mentions_how_to_enable_claude(monkeypatch):
    at = _assistant(monkeypatch, "auto")
    assert "No LLM is configured" in _all_text(at)
    assert "ANTHROPIC_API_KEY" in _all_text(at)
