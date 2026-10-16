"""
config.py
-----------
Shared paths and documented project assumptions for IntelliGrid AI.

Kept as a single source of truth so later phases (wastage detection,
anomaly detection, optimization, ...) reuse the same constants instead of
redefining them, and so every "assumption vs. measured value" distinction
required by the project's research-integrity rules lives in one reviewable
place.

Nothing in this file alters data. It only names paths and documents fixed,
already-approved Phase 0 decisions (see TRANSFORM_LOG.md for the full
justification of each).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
ORIGINAL_DATASET_PATH = DATA_DIR / "original" / "intelligrid_energy_dataset_with_4_fields.csv"
PROCESSED_DATASET_PATH = DATA_DIR / "processed" / "intelligrid_energy_dataset.csv"
ROOM_REFERENCE_PATH = DATA_DIR / "processed" / "room_reference.csv"

# ---------------------------------------------------------------------------
# Electricity tariff time-of-use bands
# -----------------------------------------------------------------------
# These bands were already present in, and verified against, the Phase 0
# source file (see TRANSFORM_LOG.md section 4: confirmed 100% consistent
# with `hour` for all 1000 original rows, zero exceptions). They are
# reproduced here, not reinvented, and must not be changed without updating
# TRANSFORM_LOG.md.
# ---------------------------------------------------------------------------
OFF_PEAK_HOURS = {0, 1, 2, 3, 4, 5, 23}
STANDARD_HOURS = set(range(6, 18))   # 6-17 inclusive
PEAK_HOURS = {18, 19, 20, 21, 22}

TARIFF_CATEGORY_BY_HOUR = {}
for _h in OFF_PEAK_HOURS:
    TARIFF_CATEGORY_BY_HOUR[_h] = "Off-Peak"
for _h in STANDARD_HOURS:
    TARIFF_CATEGORY_BY_HOUR[_h] = "Standard"
for _h in PEAK_HOURS:
    TARIFF_CATEGORY_BY_HOUR[_h] = "Peak"

# Independent cross-check: the specific tariff *rate* values observed in the
# processed dataset, mapped to the category they should belong to. Used only
# to flag a mismatch for investigation -- never to overwrite the hour-based
# classification above, which is the documented source of truth.
TARIFF_RATE_TO_CATEGORY = {
    5.0: "Off-Peak",
    7.0: "Standard",
    9.0: "Peak",
}

# ---------------------------------------------------------------------------
# Temperature descriptive bands -- EXPLICIT PROJECT ASSUMPTION
# -----------------------------------------------------------------------
# These thresholds are NOT an engineering or comfort standard (e.g. not
# ASHRAE 55). They exist only to give a human-readable label alongside the
# numeric temperature value for descriptive/reporting convenience. Phase 1
# never uses temperature_context in any classification or wastage-indicator
# logic -- only the numeric `temperature` field and HVAC/lighting/occupancy
# states are used for that.
# ---------------------------------------------------------------------------
TEMPERATURE_COOL_MAX = 22.0      # temperature < this -> "Cool"
TEMPERATURE_MODERATE_MAX = 26.0  # this <= temperature <= TEMPERATURE_MODERATE_MAX -> "Moderate"
                                   # temperature > TEMPERATURE_MODERATE_MAX -> "Warm"

# ---------------------------------------------------------------------------
# Required fields for condition classification (Phase 1)
# ---------------------------------------------------------------------------
REQUIRED_CLASSIFICATION_FIELDS = ["occupancy", "hvac_usage", "lighting_usage"]
VALID_STATUS_VALUES = {"On", "Off"}

# ---------------------------------------------------------------------------
# Context-aware anomaly detection (Phase 3) -- documented statistical design
# choices, not physical energy-efficiency thresholds. See
# modules/anomaly_detection.py and README.md for full justification.
# ---------------------------------------------------------------------------
MIN_CONTEXT_GROUP_SIZE = 5     # groups with fewer observations are marked INSUFFICIENT_BASELINE, never classified
ANOMALY_Z_THRESHOLD = 2.0      # |z_score| >= this -> HIGH/LOW_ENERGY_ANOMALY; a statistical, not physical, threshold
ANOMALY_STD_EPSILON = 1e-9     # context_std below this is treated as zero/unusable (numerical safety, not statistical)

# ---------------------------------------------------------------------------
# RAG (Phase 7) -- see rag/embeddings.py for why sentence-transformers is an
# auto-detected optional upgrade rather than a hard requirement in this
# environment (disk-space constrained).
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # used only if sentence-transformers is importable
RAG_TOP_K = 3                                 # number of chunks retrieved per query
RAG_SIMILARITY_THRESHOLD = 0.05               # below this score, retrieval is treated as "no relevant evidence found"

# ---------------------------------------------------------------------------
# LLM provider (Phase 8) -- no key required by default; see llm/provider.py.
#
# LLM_PROVIDER (override with the LLM_PROVIDER environment variable):
#   "auto"      (default) -> "anthropic" if an Anthropic API key is available,
#                            otherwise "null" (retrieval-only, no key needed)
#   "null"      -> never call an LLM; show retrieved passages, labelled as such
#   "anthropic" -> Claude via the official `anthropic` SDK (ANTHROPIC_API_KEY)
#   "ollama" | "openai" -> as before
#
# API keys are NEVER stored here. They are read from the environment
# (ANTHROPIC_API_KEY / OPENAI_API_KEY), from a git-ignored .env file, or from
# Streamlit secrets (.streamlit/secrets.toml, also git-ignored).
# ---------------------------------------------------------------------------
import os as _os

try:  # optional: load a git-ignored .env file if python-dotenv is installed
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(BASE_DIR / ".env", override=False)  # real environment variables win
except ImportError:
    pass

LLM_PROVIDER = (_os.getenv("LLM_PROVIDER") or "auto").strip().lower()
LLM_MODEL_NAME = "gpt-3.5-turbo"   # used by OpenAIProvider only

# Claude model identifier. Override with the ANTHROPIC_MODEL environment variable
# so a model can be changed or retired without editing code.
ANTHROPIC_MODEL_NAME = "claude-sonnet-5-5"
ANTHROPIC_MAX_TOKENS = 700           # upper bound on the length of a generated answer
ANTHROPIC_TIMEOUT_SECONDS = 30.0     # per-request timeout; the app falls back to retrieved text on failure
