"""
experiments/rag_evaluation.py
----------------------------------
EXPERIMENT 4 -- RAG Grounding Evaluation (RQ4: "Does RAG improve the
grounding of AI-generated energy explanations?").

Uses a small, MANUALLY-VERIFIED question set (written and labeled by the
project author -- each question's `expected_topic_keyword` was chosen by
reading rag/knowledge_base.py's actual document topics, not invented after
the fact) to measure retrieval precision@1 / precision@3 and the grounded
response rate. Also includes deliberately off-topic questions to measure
whether the pipeline correctly reports "no evidence" rather than
fabricating an answer.

Scores reported here are computed directly from this run -- nothing is
pre-filled or fabricated.
"""

from typing import Dict, List

from rag.rag_pipeline import answer_with_rag
from rag.retriever import retrieve_relevant_chunks

# Each on-topic question is manually matched to the topic keyword of the
# document it SHOULD retrieve, based on rag/knowledge_base.py's actual
# corpus (verified by reading the corpus, not guessed).
EVALUATION_QUESTIONS = [
    {"question": "Why should HVAC be reduced when a room is unoccupied?", "expected_topic_keyword": "hvac", "on_topic": True},
    {"question": "What are common ways to reduce HVAC energy usage?", "expected_topic_keyword": "hvac", "on_topic": True},
    {"question": "Why should lighting be turned off in unoccupied rooms?", "expected_topic_keyword": "lighting", "on_topic": True},
    {"question": "How does occupancy affect HVAC management?", "expected_topic_keyword": "occupancy", "on_topic": True},
    {"question": "What is a phantom or standby load for an appliance?", "expected_topic_keyword": "appliance", "on_topic": True},
    {"question": "How can building energy consumption be reduced overall?", "expected_topic_keyword": "conservation", "on_topic": True},
    {"question": "What temperature range is typically considered comfortable indoors?", "expected_topic_keyword": "comfort", "on_topic": True},
    {"question": "What is a peak electricity tariff period?", "expected_topic_keyword": "tariff", "on_topic": True},
    {"question": "What is the capital of France?", "expected_topic_keyword": None, "on_topic": False},
    {"question": "What is the best recipe for chocolate cake?", "expected_topic_keyword": None, "on_topic": False},
]


def _topic_matches(chunk: Dict, keyword: str) -> bool:
    return keyword.lower() in chunk["topic"].lower()


def run_experiment() -> Dict:
    results: List[Dict] = []
    on_topic_hits_at_1 = 0
    on_topic_hits_at_3 = 0
    on_topic_total = 0
    off_topic_correctly_reported_no_evidence = 0
    off_topic_total = 0
    grounded_count = 0

    for item in EVALUATION_QUESTIONS:
        question = item["question"]
        retrieved = retrieve_relevant_chunks(question, top_k=3)
        rag_result = answer_with_rag(question)

        entry = {
            "question": question,
            "on_topic": item["on_topic"],
            "expected_topic_keyword": item["expected_topic_keyword"],
            "retrieved_titles": [r["title"] for r in retrieved],
            "mode": rag_result["mode"],
            "grounded": rag_result["grounded"],
        }

        if item["on_topic"]:
            on_topic_total += 1
            hit_at_1 = bool(retrieved) and _topic_matches(retrieved[0], item["expected_topic_keyword"])
            hit_at_3 = any(_topic_matches(r, item["expected_topic_keyword"]) for r in retrieved)
            entry["hit_at_1"] = hit_at_1
            entry["hit_at_3"] = hit_at_3
            on_topic_hits_at_1 += int(hit_at_1)
            on_topic_hits_at_3 += int(hit_at_3)
            if rag_result["grounded"]:
                grounded_count += 1
        else:
            off_topic_total += 1
            correctly_reported = rag_result["mode"] == "no_evidence"
            entry["correctly_reported_no_evidence"] = correctly_reported
            off_topic_correctly_reported_no_evidence += int(correctly_reported)

        results.append(entry)

    return {
        "label": "Manually-verified question set; scores computed directly from this run, not pre-filled.",
        "n_on_topic_questions": on_topic_total,
        "n_off_topic_questions": off_topic_total,
        "precision_at_1_on_topic": round(on_topic_hits_at_1 / on_topic_total, 3) if on_topic_total else None,
        "precision_at_3_on_topic": round(on_topic_hits_at_3 / on_topic_total, 3) if on_topic_total else None,
        "grounded_response_rate_on_topic": round(grounded_count / on_topic_total, 3) if on_topic_total else None,
        "off_topic_correct_no_evidence_rate": (
            round(off_topic_correctly_reported_no_evidence / off_topic_total, 3) if off_topic_total else None
        ),
        "per_question_results": results,
        "embedding_method_caveat": (
            "Retrieval uses a TF-IDF fallback (keyword-overlap based), not a neural sentence "
            "embedding -- see rag/embeddings.py. Precision at this corpus size/method combination "
            "should not be assumed to generalize to a larger or more semantically diverse corpus."
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_experiment(), indent=2, default=str))
