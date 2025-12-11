import openai 
import argparse
import json
import os
import time
import math
import json
import re
from typing import List, Dict, Any, Tuple

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import sentence-transformers and transformers
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Optional: openai (only if user wants to generate a response via OpenAI)
try:
    import openai
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

# Download punkt for sentence splitting if needed
# Download required NLTK tokenizers
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)





def load_json(path):
    """
    Loads JSON but also auto-fixes common formatting issues:
    - trailing commas
    - malformed endings
    - accidental extra commas inside objects/arrays
    """

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # --- Auto-fix 1: remove trailing commas before } or ] ---
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    # --- Auto-fix 2: remove BOM markers (Windows Notepad issue) ---
    raw = raw.lstrip("\ufeff")

    # --- Auto-fix 3: collapse invalid multiple commas ---
    raw = re.sub(r",\s*,", ",", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print("\n❌ JSON is still invalid after auto-fix:")
        print(f"Error: {str(e)}")
        print(f"File: {path}")
        print("\n🔎 Please paste the JSON here for manual repair.\n")
        raise e



def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in nltk.tokenize.sent_tokenize(text) if s.strip()]


class Evaluator:
    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        nli_model_name: str = "facebook/bart-large-mnli",
        relevance_threshold: float = 0.55,
        completeness_threshold: float = 0.55,
        hallucination_similarity_threshold: float = 0.45,
    ):
        # Embedding model (sentence-transformers)
        self.embed_model = SentenceTransformer(embed_model_name)
        # NLI pipeline (entailment / contradiction / neutral)
        self.nli = pipeline("text-classification", model=nli_model_name, return_all_scores=True)
        # thresholds
        self.relevance_threshold = relevance_threshold
        self.completeness_threshold = completeness_threshold
        self.hallucination_similarity_threshold = hallucination_similarity_threshold

def embed_texts(self, texts):
    # Ensure input is always a list
    if isinstance(texts, str):
        texts = [texts]
        texts = [str(t) for t in texts]
        embeddings = self.embed_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

def compute_relevance_score(self,wsresponse_text: str,user_query: str,context_texts: List[str],) -> Dict[str, Any]:
    texts = [response_text, user_query] + context_texts
    emb = self.embed_texts(texts)
    resp_emb = emb[0:1]
    user_emb = emb[1:2]
    context_embs = emb[2:]
    sim_to_query = float(cosine_similarity(resp_emb, user_emb)[0, 0])
    if len(context_embs) > 0:
        sims_to_context = cosine_similarity(resp_emb, context_embs)[0]
        max_sim = float(np.max(sims_to_context))
        mean_sim = float(np.mean(sims_to_context))
    else:
        max_sim = 0.0
        mean_sim = 0.0

        # Composite relevance score (weighted)
        relevance_score = 0.5 * sim_to_query + 0.5 * max_sim

        return {
            "similarity_to_query": sim_to_query,
            "max_similarity_to_context": max_sim,
            "mean_similarity_to_context": mean_sim,
            "relevance_score": relevance_score,
            "is_relevant": relevance_score >= self.relevance_threshold,
        }

    def compute_completeness(
        self,
        response_text: str,
        context_texts: List[str],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Completeness: measure what fraction of the *top-k* relevant context chunks
        are reflected in the response. We compute embeddings and mark a context chunk
        as 'covered' if similarity(response, context_chunk) >= completeness_threshold.
        """
        if len(context_texts) == 0:
            return {"coverage_fraction": 0.0, "covered_chunks": [], "uncovered_chunks": []}

        resp_emb = self.embed_texts([response_text])
        ctx_embs = self.embed_texts(context_texts)
        sims = cosine_similarity(resp_emb, ctx_embs)[0]
        # pick top-k most relevant context chunks
        idx_sorted = np.argsort(-sims)
        top_idx = idx_sorted[:min(top_k, len(context_texts))]
        covered = []
        uncovered = []
        for i in top_idx:
            if sims[i] >= self.completeness_threshold:
                covered.append({"index": int(i), "text": context_texts[i], "similarity": float(sims[i])})
            else:
                uncovered.append({"index": int(i), "text": context_texts[i], "similarity": float(sims[i])})
        coverage_fraction = len(covered) / len(top_idx)
        return {"coverage_fraction": coverage_fraction, "covered_chunks": covered, "uncovered_chunks": uncovered}

    def detect_hallucinations_and_accuracy(
        self,
        response_text: str,
        context_texts: List[str],
    ) -> Dict[str, Any]:
        """
        For each sentence in the response:
          - find best-matching context chunk by cosine similarity
          - run NLI between context_chunk (premise) and response_sentence (hypothesis)
          - if best similarity < threshold OR NLI label is 'contradiction' or 'neutral', flag sentence
            as 'unsupported' (possible hallucination). If 'entailment' and similarity high -> supported.
        Returns per-sentence analysis and an aggregate hallucination score.
        """
        sentences = split_sentences(response_text)
        if len(sentences) == 0:
            return {"sentences": [], "hallucinated_sent_count": 0, "hallucination_rate": 0.0}

        sent_embs = self.embed_texts(sentences)
        ctx_embs = self.embed_texts(context_texts) if len(context_texts) > 0 else np.zeros((0, sent_embs.shape[1]))

        analysis = []
        hallucinated_count = 0
        for i, sent in enumerate(sentences):
            best_sim = 0.0
            best_ctx_idx = None
            best_ctx_text = ""
            if ctx_embs.shape[0] > 0:
                sims = cosine_similarity(sent_embs[i : i + 1], ctx_embs)[0]
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_ctx_idx = best_idx
                best_ctx_text = context_texts[best_idx]

            # Default: if no context at all, mark as unsupported
            if best_ctx_idx is None:
                label = "no_context"
                score = None
                supported = False
            else:
                # NLI between premise (context chunk) and hypothesis (answer sentence)
                nli_input = f"{best_ctx_text}\n\nHypothesis: {sent}"
                # Use the pipeline; returns list of dicts each with label and score
                nli_scores = self.nli({"text": best_ctx_text}, candidate_labels=[sent], hypothesis_template=None)
                # The transformers NLI pipeline with return_all_scores True returns a list of label-score dicts.
                # But using pipeline like above can differ. To be robust, we'll use the classic approach:
                # Use entailment via calling the model with premise/hypothesis using the pipeline's text-classification with the "entailment" labels
                # Because transformer pipeline for multi-label could be inconsistent, attempt to interpret results.
                # Simpler: run model in pairwise NLI format: use sequence pair classification via `self.nli` by passing
                # the premise and hypothesis in the usual way if supported. Fallback: if pipeline returns odd format, mark as neutral.
                try:
                    # transformers' "text-classification" with bart-large-mnli expects inputs differently.
                    # The HuggingFace pipeline allows passing `sequence_pair` as tuple sometimes; but we handle both.
                    res = self.nli(f"{best_ctx_text} </s></s> {sent}")
                    # res is a list of lists with dicts
                    # Attempt to find 'ENTAILMENT' score
                    flat = res[0] if isinstance(res, list) and len(res) > 0 else res
                    # convert to mapping
                    lbl_scores = {}
                    for d in flat:
                        lbl = d.get("label", "").lower()
                        lbl_scores[lbl] = d.get("score", 0.0)
                    # choose highest label
                    if lbl_scores:
                        # common labels might be 'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL' (case may vary)
                        # choose best
                        best_label = max(lbl_scores.items(), key=lambda kv: kv[1])[0]
                        best_label_score = float(lbl_scores[best_label])
                        label = best_label
                        score = best_label_score
                    else:
                        label = "unknown"
                        score = None
                except Exception:
                    # Fallback: simple heuristic from similarity
                    label = "unknown"
                    score = None

                # Heuristic: mark supported if label indicates entailment OR similarity high
                supported = False
                if label and ("entail" in label):
                    supported = True
                elif best_sim >= self.hallucination_similarity_threshold:
                    supported = True
                else:
                    supported = False

            if not supported:
                hallucinated_count += 1

            analysis.append(
                {
                    "sentence": sent,
                    "best_context_index": best_ctx_idx,
                    "best_context_text": best_ctx_text,
                    "similarity_to_best_context": best_sim,
                    "nli_label": label,
                    "nli_score": score,
                    "supported_by_context": supported,
                }
            )

        hallucination_rate = hallucinated_count / len(sentences)
        return {"sentences": analysis, "hallucinated_sent_count": hallucinated_count, "hallucination_rate": hallucination_rate}

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """
        Heuristic token estimate: average 4 characters per token (rough).
        Replace with tiktoken or model-specific tokenizer for production.
        """
        if not text:
            return 0
        chars = len(text)
        est = max(1, int(chars / 4))
        return est

class Evaluator:
    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        nli_model_name: str = "facebook/bart-large-mnli",
        relevance_threshold: float = 0.55,
        completeness_threshold: float = 0.55,
        hallucination_similarity_threshold: float = 0.45,
    ):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.nli = pipeline("text-classification", model=nli_model_name, top_k=None)
        self.relevance_threshold = relevance_threshold
        self.completeness_threshold = completeness_threshold
        self.hallucination_similarity_threshold = hallucination_similarity_threshold

    # -----------------------------
    # FIXED: embed_texts (correct indentation + logic)
    # -----------------------------
    def embed_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        texts = [str(t) for t in texts]

        embeddings = self.embed_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    # -----------------------------
    # FIXED: relevance score
    # -----------------------------
    def compute_relevance_score(
        self,
        response_text: str,
        user_query: str,
        context_texts: List[str],
    ) -> Dict[str, Any]:

        texts = [response_text, user_query] + context_texts
        emb = self.embed_texts(texts)

        resp_emb = emb[0:1]
        user_emb = emb[1:2]
        context_embs = emb[2:]

        sim_to_query = float(cosine_similarity(resp_emb, user_emb)[0, 0])

        if len(context_embs) > 0:
            sims_to_context = cosine_similarity(resp_emb, context_embs)[0]
            max_sim = float(np.max(sims_to_context))
            mean_sim = float(np.mean(sims_to_context))
        else:
            max_sim = 0.0
            mean_sim = 0.0

        relevance_score = 0.5 * sim_to_query + 0.5 * max_sim

        return {
            "similarity_to_query": sim_to_query,
            "max_similarity_to_context": max_sim,
            "mean_similarity_to_context": mean_sim,
            "relevance_score": relevance_score,
            "is_relevant": relevance_score >= self.relevance_threshold,
        }

    # -----------------------------
    # FIXED: completeness
    # -----------------------------
    def compute_completeness(
        self,
        response_text: str,
        context_texts: List[str],
        top_k: int = 5
    ):
        if len(context_texts) == 0:
            return {"coverage_fraction": 0.0, "covered_chunks": [], "uncovered_chunks": []}

        resp_emb = self.embed_texts([response_text])
        ctx_embs = self.embed_texts(context_texts)

        sims = cosine_similarity(resp_emb, ctx_embs)[0]
        idx_sorted = np.argsort(-sims)
        top_idx = idx_sorted[:min(top_k, len(context_texts))]

        covered = []
        uncovered = []

        for i in top_idx:
            if sims[i] >= self.completeness_threshold:
                covered.append({"index": int(i), "text": context_texts[i], "similarity": float(sims[i])})
            else:
                uncovered.append({"index": int(i), "text": context_texts[i], "similarity": float(sims[i])})

        return {
            "coverage_fraction": len(covered) / len(top_idx),
            "covered_chunks": covered,
            "uncovered_chunks": uncovered,
        }

    # -----------------------------
    # FIXED: hallucination detection
    # -----------------------------
    def detect_hallucinations_and_accuracy(
        self,
        response_text: str,
        context_texts: List[str]
    ):
        sentences = split_sentences(response_text)

        if len(sentences) == 0:
            return {"sentences": [], "hallucinated_sent_count": 0, "hallucination_rate": 0.0}

        sent_embs = self.embed_texts(sentences)
        ctx_embs = self.embed_texts(context_texts) if len(context_texts) else np.zeros((0, sent_embs.shape[1]))

        analysis = []
        hallucinated = 0

        for i, sent in enumerate(sentences):
            if ctx_embs.shape[0] > 0:
                sims = cosine_similarity(sent_embs[i:i+1], ctx_embs)[0]
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_ctx_text = context_texts[best_idx]
            else:
                best_idx = None
                best_sim = 0.0
                best_ctx_text = ""

            label = "unknown"
            score = None
            supported = False

            if best_idx is not None:
                try:
                    res = self.nli(f"{best_ctx_text} </s></s> {sent}")
                    flat = res[0]
                    lbl_scores = {d["label"].lower(): d["score"] for d in flat}
                    best_label = max(lbl_scores.items(), key=lambda kv: kv[1])[0]
                    label = best_label
                    score = lbl_scores[best_label]
                except Exception:
                    label = "unknown"

                if "entail" in label or best_sim >= self.hallucination_similarity_threshold:
                    supported = True

            if not supported:
                hallucinated += 1

            analysis.append({
                "sentence": sent,
                "best_context_index": best_idx,
                "best_context_text": best_ctx_text,
                "similarity_to_best_context": best_sim,
                "nli_label": label,
                "nli_score": score,
                "supported_by_context": supported
            })

        return {
            "sentences": analysis,
            "hallucinated_sent_count": hallucinated,
            "hallucination_rate": hallucinated / len(sentences),
        }

    # -----------------------------
    # Token estimation
    # -----------------------------
    @staticmethod
    def estimate_token_count(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    # -----------------------------
    # FIXED: evaluate method
    # -----------------------------
    def evaluate(
        self,
        conversation: Dict[str, Any],
        context_vectors: List[Dict[str, Any]],
        response_text: str,
        measure_time: bool = True
    ):
        if isinstance(conversation, dict):
            msgs = conversation.get("messages", conversation.get("conversation", []))
        else:
            msgs = conversation

        user_query = ""
        for m in reversed(msgs):
            if m.get("role", "").lower() in ("user", "human"):
                user_query = m.get("content", "")
                break

        context_texts = [str(c.get("text", "")) for c in context_vectors]

        t0 = time.monotonic()
        rel = self.compute_relevance_score(response_text, user_query, context_texts)
        t1 = time.monotonic()

        comp = self.compute_completeness(response_text, context_texts)
        t2 = time.monotonic()

        hall = self.detect_hallucinations_and_accuracy(response_text, context_texts)
        t3 = time.monotonic()

        all_text = response_text + user_query + " ".join(context_texts)
        est_tokens = self.estimate_token_count(all_text)
        cost_usd = (est_tokens / 1000) * 0.02

        return {
            "results": {
                "relevance": rel,
                "completeness": comp,
                "hallucination": hall,
                "timings_ms": {
                    "relevance_ms": (t1 - t0) * 1000,
                    "completeness_ms": (t2 - t1) * 1000,
                    "hallucination_ms": (t3 - t2) * 1000,
                    "total_eval_ms": (t3 - t0) * 1000,
                },
                "estimated_tokens": est_tokens,
                "estimated_cost_usd": cost_usd,
            }
        }



def generate_response_with_openai(conversation: Dict[str, Any], context_vectors: List[Dict[str, Any]], model="gpt-4o") -> Tuple[str, Dict[str, Any]]:
    """
    Simple composition: take top-k context texts, prepend to the prompt, and call OpenAI's chat completion.
    Requires OPENAI_API_KEY env var. Returns (response_text, metadata).
    NOTE: This function is simple and intended for demo. For production you'd create better prompt templates,
    chunking strategies, and safety filters.
    """
    assert HAVE_OPENAI, "openai package is not installed"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    openai.api_key = api_key

    # Compose system prompt + context
    top_ctx_texts = [c.get("text", "") for c in context_vectors[:5]]
    system_prompt = (
        "You are an assistant. Use the context below to answer. If the context does not contain the "
        "information, answer concisely and say you don't have that info.\n\nContext:\n"
        + "\n\n---\n\n".join(top_ctx_texts)
    )

    # Build messages from conversation
    conv_msgs = []
    # put system prompt first
    conv_msgs.append({"role": "system", "content": system_prompt})
    if isinstance(conversation, dict):
        msgs = conversation.get("messages", conversation.get("conversation", []))
    else:
        msgs = conversation
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "") or m.get("text", "")
        if role and content:
            conv_msgs.append({"role": role, "content": content})

    t0 = time.time()
    completion = openai.ChatCompletion.create(model=model, messages=conv_msgs, max_tokens=512)
    t1 = time.time()
    response_text = completion["choices"][0]["message"]["content"]
    metadata = {"openai_elapsed_s": t1 - t0, "openai_usage": completion.get("usage", {})}
    return response_text, metadata


def main():
    parser = argparse.ArgumentParser(description="LLM response evaluation pipeline")
    parser.add_argument("--conv", required=True, help="Path to conversation JSON")
    parser.add_argument("--ctx", required=True, help="Path to context_vectors JSON")
    parser.add_argument("--response_file", required=False, help="Path to file containing LLM response to evaluate")
    parser.add_argument("--call_openai", action="store_true", help="If provided, call OpenAI to generate response (requires OPENAI_API_KEY)")
    parser.add_argument("--out", default="report.json", help="Output report JSON file")
    args = parser.parse_args()

    conversation = load_json(args.conv)
    context_vectors = load_json(args.ctx)
    # Expect context_vectors to be a list of objects with 'text' field. If it's a dict with 'results', adapt.
    if isinstance(context_vectors, dict):
        # try to find likely places
        if "results" in context_vectors and isinstance(context_vectors["results"], list):
            context_vectors = context_vectors["results"]
        elif "contexts" in context_vectors:
            context_vectors = context_vectors["contexts"]
        else:
            # fallback: if it's a mapping of id->text, turn into list
            context_vectors = [{"text": v} for v in context_vectors.values()]

    response_text = ""
    extra_meta = {}
    if args.call_openai:
        if not HAVE_OPENAI:
            raise RuntimeError("openai package not installed; install openai to call OpenAI API")
        response_text, meta = generate_response_with_openai(conversation, context_vectors)
        extra_meta["llm_generation"] = meta
    elif args.response_file:
        with open(args.response_file, "r", encoding="utf-8") as f:
            response_text = f.read()
    else:
        raise RuntimeError("Either provide --response_file with response text or use --call_openai")

    evaluator = Evaluator()
    report = evaluator.evaluate(conversation, context_vectors, response_text)
    report["input_summary"] = {
        "conversation_messages": len(conversation.get("messages", conversation.get("conversation", []))) if isinstance(conversation, dict) else len(conversation),
        "context_chunks": len(context_vectors),
        "response_len_chars": len(response_text),
    }
    report["llm_response"] = {"text": response_text[:1000] + ("..." if len(response_text) > 1000 else "")}
    report["extra_meta"] = extra_meta

    save_json(args.out, report)
    print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
