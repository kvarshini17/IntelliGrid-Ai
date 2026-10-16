"""
pages/about.py
-----------------
Purpose : About page -- research-integrity notes, limitations, and
          assumptions, kept visible in the running application (not just in
          README.md) so a reviewer can find them without reading source code.
"""

import streamlit as st


def render():
    st.header("About IntelliGrid AI")
    st.markdown("""
**IntelliGrid AI** -- Explainable Energy Consumption Optimization for Smart Buildings.

A software-only research prototype using simulated building-energy data. There is no
physical IoT hardware. Deterministic analytical modules (condition, wastage, anomaly,
optimization) are the source of truth; the LLM/RAG layer only explains their results --
it never overrides them.

### Important distinctions
- **Condition** ("what is happening?") -- Phase 1.
- **Potential wastage** ("could this operating condition represent unnecessary use?") --
  Phase 2, a condition-based indicator, never derived from energy_consumption magnitude.
- **Anomaly** ("is this statistically unusual for a comparable context?") -- Phase 3,
  derived purely from energy_consumption relative to a context-conditioned baseline.
  Wastage and anomaly are kept strictly independent.
- **Optimization** ("what action is feasible and preferable?") -- Phase 4, a deterministic
  candidate-scoring method (no search algorithm, no ML).
- **Recommendation** -- Phase 5, a templating layer synthesizing the above.

### Known limitations
- The dataset has no ground-truth wastage or anomaly labels. No precision/recall/F1 is
  claimed against it.
- The anomaly baseline is in-sample/offline (see README and `modules/anomaly_detection.py`),
  not a real-time detector.
- Optimization scores are ordinal priority weights, not kWh/cost estimates -- this dataset
  has no appliance-level sub-metering.
- The knowledge base is project-authored general guidance, not verified primary-source
  citations (see `rag/knowledge_base.py`'s source policy).
- The embedding method is a TF-IDF fallback in this environment (sentence-transformers
  requires a multi-GB PyTorch install not available here); FAISS itself is genuinely used
  for the vector index.
- No LLM is required for the system to function; the default Null provider returns
  retrieved knowledge text directly, clearly labeled as not LLM-generated.

### Documented assumptions
- Room square footage is fixed per room (Phase 0 room reference table).
- Room capacity is metadata/assumption only (100 sq ft/occupant); occupancy is never
  altered to fit it.
- Electricity tariff bands are a documented project assumption.
- "Holiday" is a synthetic, independently-sampled flag, not tied to a real calendar.
- Renewable energy's unit is an explicitly unverified assumption (no source documentation
  was available).
- `"None"` in the Appliance field represents "no active appliance," not missing data.
""")
