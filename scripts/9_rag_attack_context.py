"""
RAG layer: grounds each CNN-predicted attack category in the real MITRE
ATT&CK knowledge base, then uses a local LLM to turn that into an
analyst-facing explanation -- instead of asking an LLM to explain "what a
PortScan is" from its own (possibly wrong, possibly hallucinated) training
knowledge.

Design decisions:

    1. Retrieval corpus: the full MITRE ATT&CK Enterprise matrix (STIX,
       official `mitre/cti` GitHub repo), not a hand-picked subset of
       techniques. A hand-picked list would be faster to build but is
       arbitrary and undermines the point of doing retrieval at all --
       letting semantic search find the relevant techniques is what makes
       this RAG rather than a lookup table.

    2. Retrieval, not fine-tuning: this project already has one deep model
       (the CNN); adding a second one (fine-tuned LLM) for something that
       plain semantic search + a general instruction-tuned model handles
       fine would be unnecessary complexity and unnecessary GPU time for
       the scope of this course project.

    3. Embeddings: sentence-transformers' `all-MiniLM-L6-v2` -- small,
       CPU-friendly, and the ATT&CK Enterprise corpus is only a few hundred
       techniques, so a full brute-force cosine similarity search is more
       than fast enough; no vector database needed.

    4. Generation: Qwen2.5-3B-Instruct, run locally (transformers), not an
       API. Apache-2.0 licensed (no gated access / HF token needed unlike
       Llama), free (no per-token cost, per the project's decision to avoid
       API billing), and small enough to run on Colab's free/purchased GPU
       tier -- or CPU, slower but workable for a handful of explanations.
       See the Work log's RAG section for the cost comparison against the
       Claude/OpenAI API that led to this choice.

    5. Grounding, not free generation: the LLM prompt explicitly instructs
       the model to use ONLY the retrieved ATT&CK technique text and to
       cite technique IDs (e.g. T1046) -- this is what makes the output
       trustworthy for a security context instead of a plausible-sounding
       hallucination.

    6. BENIGN is excluded from RAG_CLASS_QUERIES (config.py) on purpose:
       there's no attack to explain, so it's skipped before this script
       ever runs, not given an empty/awkward explanation.

Pipeline position:
    models/cnn_resnet18_best.pt (predicted class) + MITRE ATT&CK STIX data
    -> [THIS SCRIPT] -> data/processed/reports/rag_attack_explanations.json

Usage:
    python 9_rag_attack_context.py                  # build index once, explain all 10 attack classes
    python 9_rag_attack_context.py --classes PortScan Bot   # just these classes (faster iteration)
    python 9_rag_attack_context.py --rebuild-index   # force re-download/re-embed ATT&CK data
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import requests

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ATTACK_DIR = config.ATTACK_DIR
ATTACK_STIX_URL = config.ATTACK_STIX_URL
ATTACK_STIX_PATH = config.ATTACK_STIX_PATH
ATTACK_TECHNIQUES_PATH = config.ATTACK_TECHNIQUES_PATH
ATTACK_EMBEDDINGS_PATH = config.ATTACK_EMBEDDINGS_PATH

RAG_EMBEDDING_MODEL = config.RAG_EMBEDDING_MODEL
RAG_LLM_MODEL = config.RAG_LLM_MODEL
RAG_TOP_K = config.RAG_TOP_K
RAG_EXPLANATIONS_PATH = config.RAG_EXPLANATIONS_PATH
RAG_CLASS_QUERIES = config.RAG_CLASS_QUERIES

REPORTS_DIR = config.REPORTS_DIR


# --------------------------------------------------------------------------- #
# Step 1: download + parse MITRE ATT&CK (retrieval corpus)
# --------------------------------------------------------------------------- #


def download_attack_stix() -> None:
    """Downloads the official MITRE ATT&CK Enterprise STIX bundle if not
    already cached locally. ~50 MB, one-time cost."""
    if os.path.exists(ATTACK_STIX_PATH):
        print(f"  ATT&CK STIX bundle already cached: {ATTACK_STIX_PATH}")
        return
    print(f"  Downloading MITRE ATT&CK Enterprise matrix from {ATTACK_STIX_URL} ...")
    response = requests.get(ATTACK_STIX_URL, timeout=120)
    response.raise_for_status()
    os.makedirs(ATTACK_DIR, exist_ok=True)
    with open(ATTACK_STIX_PATH, "wb") as f:
        f.write(response.content)
    print(f"  Saved to {ATTACK_STIX_PATH} ({len(response.content) / 1024**2:.1f} MB)")


# The 14 real MITRE ATT&CK Enterprise tactic names (phase_name values), used
# as a defensive whitelist below. Filtering kill_chain_phases to
# kill_chain_name == "mitre-attack" (already done below) is normally enough,
# but at least one technique in the STIX bundle (T1564.006) was observed to
# carry a stale/non-standard phase_name ("stealth") under that same
# kill_chain_name -- a known quirk of the STIX export, not something this
# script can fix upstream. Whitelisting the real tactic names catches that
# case (and any similar one) instead of silently saving a fabricated-looking
# tactic label into the retrieval corpus.
VALID_ENTERPRISE_TACTICS = {
    "reconnaissance", "resource-development", "initial-access", "execution",
    "persistence", "privilege-escalation", "defense-evasion",
    "credential-access", "discovery", "lateral-movement", "collection",
    "command-and-control", "exfiltration", "impact",
}


def parse_attack_techniques() -> list[dict]:
    """Extracts (technique_id, name, description, tactics) for every
    non-deprecated, non-revoked technique ('attack-pattern' STIX objects)
    in the Enterprise matrix. Returns the full technique list -- no
    hand-picked filtering by tactic, so semantic retrieval later does the
    actual relevance judgment instead of a hardcoded guess."""
    with open(ATTACK_STIX_PATH, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    techniques = []
    dropped_tactics = set()
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        technique_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                technique_id = ref.get("external_id")
                break
        if technique_id is None:
            continue  # shouldn't happen for a real technique, but be defensive 

        tactics = []
        for phase in obj.get("kill_chain_phases", []):
            if phase.get("kill_chain_name") != "mitre-attack":
                continue
            phase_name = phase.get("phase_name")
            if phase_name in VALID_ENTERPRISE_TACTICS:
                tactics.append(phase_name)
            elif phase_name:
                dropped_tactics.add(phase_name)

        techniques.append(
            {
                "technique_id": technique_id,
                "name": obj.get("name", ""),
                "description": (obj.get("description", "") or "").split("\n")[0],
                # first paragraph only -- ATT&CK descriptions often run to
                # several paragraphs with markdown/citation clutter that
                # adds tokens without adding retrieval signal
                "tactics": tactics,
            }
        )

    if dropped_tactics:
        print(
            f"  Note: dropped {len(dropped_tactics)} non-standard tactic label(s) "
            f"found in the STIX bundle (not real Enterprise tactic names): "
            f"{sorted(dropped_tactics)}"
        )

    return techniques


def build_attack_index(rebuild: bool = False) -> tuple[list[dict], np.ndarray]:
    """Ensures the ATT&CK technique list + embeddings exist on disk (building
    them once, ~1-2 minutes on CPU for the ~600-technique Enterprise corpus),
    then loads and returns them."""
    have_cache = (
        os.path.exists(ATTACK_TECHNIQUES_PATH) and os.path.exists(ATTACK_EMBEDDINGS_PATH)
    )
    if have_cache and not rebuild:
        print("  Using cached ATT&CK technique index.")
        with open(ATTACK_TECHNIQUES_PATH, "r", encoding="utf-8") as f:
            techniques = json.load(f)
        embeddings = np.load(ATTACK_EMBEDDINGS_PATH)
        return techniques, embeddings

    print("Building ATT&CK technique index (first run, or --rebuild-index)...")
    download_attack_stix()
    techniques = parse_attack_techniques()
    print(f"  Parsed {len(techniques)} non-deprecated techniques.")

    from sentence_transformers import SentenceTransformer

    print(f"  Embedding technique text with {RAG_EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(RAG_EMBEDDING_MODEL)
    corpus = [f"{t['name']}. {t['description']}" for t in techniques]
    embeddings = embedder.encode(
        corpus, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )

    os.makedirs(ATTACK_DIR, exist_ok=True)
    with open(ATTACK_TECHNIQUES_PATH, "w", encoding="utf-8") as f:
        json.dump(techniques, f)
    np.save(ATTACK_EMBEDDINGS_PATH, embeddings)
    print(f"  Saved index: {ATTACK_TECHNIQUES_PATH}, {ATTACK_EMBEDDINGS_PATH}")

    return techniques, embeddings


# --------------------------------------------------------------------------- #
# Step 2: retrieval
# --------------------------------------------------------------------------- #


def retrieve_techniques(
    query: str,
    techniques: list[dict],
    embeddings: np.ndarray,
    embedder,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """Brute-force cosine similarity search -- fine at this corpus size
    (~600 techniques), no vector DB needed."""
    query_embedding = embedder.encode([query], convert_to_numpy=True)[0]

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    norms[norms == 0] = 1e-9  # guard against a zero-vector edge case
    similarities = (embeddings @ query_embedding) / norms

    top_indices = np.argsort(-similarities)[:top_k]
    return [
        {**techniques[i], "similarity": float(similarities[i])} for i in top_indices
    ]


# --------------------------------------------------------------------------- #
# Step 3: grounded generation (local LLM)
# --------------------------------------------------------------------------- #


def build_prompt(predicted_class: str, retrieved: list[dict]) -> list[dict]:
    context_block = "\n\n".join(
        f"[{t['technique_id']}] {t['name']}\n{t['description']}" for t in retrieved
    )
    system_prompt = (
        "You are a security analyst assistant. You explain a network intrusion "
        "detection system's predictions using ONLY the MITRE ATT&CK context "
        "provided below -- never invent techniques, IDs, or details not present "
        "in that context. Cite technique IDs (e.g. T1046) inline. Write 3-5 "
        "sentences, plain language suitable for a SOC analyst triaging alerts."
    )
    user_prompt = (
        f"The intrusion detection model flagged this flow as: {predicted_class}\n\n"
        f"Retrieved MITRE ATT&CK context:\n{context_block}\n\n"
        "Explain what this detection likely means and why it matters, "
        "grounded strictly in the context above."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_explanation(pipe, predicted_class: str, retrieved: list[dict]) -> str:
    messages = build_prompt(predicted_class, retrieved)
    output = pipe(messages, max_new_tokens=300, do_sample=False)
    return output[0]["generated_text"][-1]["content"].strip()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ATT&CK-grounded explanations for CNN-predicted attack classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Only generate explanations for these class names (space-separated, "
             "must match config.RAG_CLASS_QUERIES keys exactly). Default: all 10 "
             "attack classes (BENIGN is never included -- there's nothing to explain).",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force re-download and re-embed the ATT&CK corpus even if a cached "
             "index already exists on disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = args.classes if args.classes else list(RAG_CLASS_QUERIES.keys())

    unknown = [c for c in class_names if c not in RAG_CLASS_QUERIES]
    if unknown:
        print(f"Unknown class name(s), not in config.RAG_CLASS_QUERIES: {unknown}")
        print(f"Valid options: {list(RAG_CLASS_QUERIES.keys())}")
        return

    print("=" * 70)
    print("STEP 1/3: ATT&CK retrieval index")
    print("=" * 70)
    techniques, embeddings = build_attack_index(rebuild=args.rebuild_index)

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(RAG_EMBEDDING_MODEL)

    print("\n" + "=" * 70)
    print("STEP 2/3: loading local LLM (Qwen2.5-3B-Instruct)")
    print("=" * 70)
    import torch
    from transformers import pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}" + ("" if device == "cuda" else " (CPU -- generation will be slower)"))
    pipe = pipeline(
        "text-generation",
        model=RAG_LLM_MODEL,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    print("\n" + "=" * 70)
    print(f"STEP 3/3: retrieval + grounded generation for {len(class_names)} class(es)")
    print("=" * 70)
    results = {}
    for predicted_class in class_names:
        query = RAG_CLASS_QUERIES[predicted_class]
        print(f"\n[{predicted_class}] query: \"{query}\"")

        retrieved = retrieve_techniques(query, techniques, embeddings, embedder)
        for t in retrieved:
            print(f"    retrieved: [{t['technique_id']}] {t['name']} (sim={t['similarity']:.3f})")

        explanation = generate_explanation(pipe, predicted_class, retrieved)
        print(f"  Explanation: {explanation}")

        results[predicted_class] = {
            "query": query,
            "retrieved_techniques": [
                {k: v for k, v in t.items() if k != "description"} for t in retrieved
            ],
            "explanation": explanation,
        }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(RAG_EXPLANATIONS_PATH):
        with open(RAG_EXPLANATIONS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(results)
    with open(RAG_EXPLANATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"\nSaved {len(results)} explanation(s) to: {RAG_EXPLANATIONS_PATH}")


if __name__ == "__main__":
    main()
