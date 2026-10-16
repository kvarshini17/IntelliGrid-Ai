# IntelliGrid AI — AI-Powered Energy Consumption Optimization for Smart Buildings

IntelliGrid AI is a software-only research prototype that analyzes simulated smart-building energy data, identifies potential energy-wastage conditions, detects statistically unusual consumption, and generates explainable optimization recommendations.

The project combines deterministic analytical modules with a Retrieval-Augmented Generation (RAG) pipeline and an optional Large Language Model (LLM) assistant.

**Important:** The project uses simulated data, not physical IoT devices or live building sensors. Analytical modules are the authoritative source of results; the LLM only explains those results and does not override them.

## Features

- **Energy Condition Analysis:** Classifies room conditions using occupancy, HVAC, and lighting states.
- **Potential Wastage Detection:** Identifies potentially unnecessary HVAC or lighting activity in unoccupied rooms.
- **Context-Aware Anomaly Detection:** Compares energy consumption with a contextual statistical baseline.
- **Energy Optimization:** Evaluates feasible appliance actions using deterministic rules and scoring.
- **Recommendation Engine:** Combines analytical outputs into structured, explainable recommendations.
- **AI Assistant:** Retrieves relevant project knowledge and provides evidence-grounded explanations.
- **Interactive Dashboard:** Presents energy analysis, optimization results, recommendations, and assistant functionality through Streamlit.
- **Research Evaluation:** Includes experiments to evaluate analytical methods and document limitations.
- **Automated Testing:** Includes unit, integration, application, and experiment tests.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and analytical logic |
| Pandas, NumPy, SciPy | Data processing and statistical analysis |
| Streamlit | Interactive web application |
| Plotly | Data visualization |
| FAISS | Vector similarity index for retrieval |
| TF-IDF | Lightweight text representation and retrieval fallback |
| RAG | Retrieves relevant knowledge to ground assistant responses |
| Claude / Anthropic API | Optional LLM-generated explanations |
| Ollama | Optional local LLM provider |
| OpenAI API | Optional hosted LLM provider |
| Pytest | Automated testing |

Neural embeddings using `sentence-transformers` are optional. If unavailable, the retrieval pipeline uses a lightweight TF-IDF fallback with FAISS.

## How It Works

1. **Dataset Preparation:** Processes the source dataset and validates the transformed data.
2. **Energy Condition Analysis:** Classifies the current operating state of each record.
3. **Potential Wastage Detection:** Applies explicit occupancy and appliance-state rules.
4. **Anomaly Detection:** Calculates statistical deviations against contextual baselines.
5. **Energy Optimization:** Scores feasible appliance actions subject to constraints.
6. **Recommendation Generation:** Combines analytical outputs into explainable recommendations.
7. **Knowledge Base Preparation:** Loads curated, project-authored energy-management guidance.
8. **RAG and LLM Integration:** Retrieves relevant passages and optionally generates an evidence-grounded explanation.
9. **Streamlit Application:** Presents the results through interactive pages.
10. **Testing and Experiments:** Tests modules and evaluates them using controlled experimental data.
11. **Research Evaluation:** Documents research questions, experimental results, limitations, and future work.

## Project Structure

```text
intelligrid_phase0/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── TRANSFORM_LOG.md
├── RESEARCH_EVALUATION.md
├── data/
│   ├── original/
│   └── processed/
├── scripts/
│   ├── preprocess.py
│   └── validate.py
├── modules/
│   ├── data_manager.py
│   ├── energy_analysis.py
│   ├── wastage_detection.py
│   ├── anomaly_detection.py
│   ├── optimization.py
│   ├── recommendation.py
│   └── pipeline.py
├── rag/
│   ├── knowledge_base.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── rag_pipeline.py
├── llm/
│   └── provider.py
├── views/
│   ├── dashboard.py
│   ├── energy_analysis.py
│   ├── optimization.py
│   ├── ai_assistant.py
│   └── about.py
├── experiments/
│   ├── wastage_evaluation.py
│   ├── anomaly_evaluation.py
│   ├── optimization_evaluation.py
│   └── rag_evaluation.py
└── tests/
```

The `views/` directory is intentional. Do not rename it to `pages/`, because Streamlit treats a `pages/` directory as a multipage navigation structure.

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- A supported operating system with compatible Python package wheels

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd intelligrid_phase0
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use a Command Prompt terminal instead:

```bat
venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

## AI Assistant and LLM Configuration

The application can run without an LLM API key. In that mode, the assistant displays retrieved knowledge passages and labels them as not LLM-generated.

Supported provider options include:

- `null` — retrieval-only mode; no LLM API key required.
- `anthropic` — Claude through the Anthropic API.
- `ollama` — a locally running Ollama model.
- `openai` — an OpenAI-compatible hosted model provider.

The default `auto` configuration selects Claude when an Anthropic API key is available; otherwise, it uses the null provider.

### Configure Claude (Optional)

1. Obtain API access through the Anthropic Console.
2. Configure your API key as an environment variable.
3. Start the application.

**Windows PowerShell:**

```powershell
$env:ANTHROPIC_API_KEY="your-api-key"
python -m streamlit run app.py
```

Never commit real API keys, passwords, or other secrets to GitHub. Claude API billing is separate from a Claude.ai subscription.

Refer to `config.py` and the project configuration documentation for supported settings.

## Testing and Research Evaluation

Run the following commands from the project root.

### Validate the processed dataset

```bash
python scripts/validate.py
```

### Run automated tests

```bash
python -m pytest tests/ -v
```

### Run research experiments

```bash
python -m experiments.wastage_evaluation
python -m experiments.anomaly_evaluation
python -m experiments.optimization_evaluation
python -m experiments.rag_evaluation
```

The experiments are designed to run independently of the Streamlit interface. They use in-memory experimental data and should not modify the original dataset.

See `RESEARCH_EVALUATION.md` for the documented research questions, experimental results, reproducibility details, and future work. See `TRANSFORM_LOG.md` for dataset provenance, transformations, assumptions, and validation details.

## Important Limitations

- All building data is simulated; the application does not connect to physical sensors or live building systems.
- Potential wastage indicates a potentially unnecessary operating condition, not confirmed energy loss.
- Anomaly detection identifies statistical unusualness, not necessarily equipment faults or energy inefficiency.
- The anomaly baseline is an offline, in-sample method, not a real-time detector.
- Records with insufficient contextual observations are marked as having an insufficient baseline instead of being assigned a guessed result.
- Optimization scores are ordinal decision scores, not measured kWh savings or monetary savings.
- The knowledge base contains project-authored general guidance and does not claim to reproduce specific verified publications.
- The LLM provides explanations; it does not perform the authoritative condition classification, wastage detection, anomaly detection, or optimization.

## Future Work

- Evaluate alternative anomaly-detection baselines and contextual features.
- Expand controlled fault-injection experiments and research evaluation.
- Evaluate neural embeddings where computing resources permit.
- Expand and improve RAG evaluation questions and retrieval quality.
- Revisit optimization scoring if appliance-level energy measurements become available.
- Explore integration with real building data and IoT systems in a future version.

## Project Status

The project documentation describes the implementation across 11 phases, from dataset preparation through research evaluation. Consult `RESEARCH_EVALUATION.md` for measured experimental findings rather than assuming the prototype demonstrates real-world energy savings.

---

**IntelliGrid AI** — Explainable, research-oriented energy analysis for simulated smart buildings.
