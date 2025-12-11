# 📘 LLM Evaluation Pipeline

This repository contains a lightweight and modular **LLM Response Evaluation Pipeline** designed to evaluate AI model responses for **relevance**, **completeness**, **hallucination detection**, **latency**, and **cost estimation**.

---

## 1. Local setup instructions:
### Clone the Repo
```bash
git clone https://github.com/itsnandu/llm-eval-pipeline.git
cd llm-eval-pipeline
```
### Install dependencies
```bash
pip install sentence-transformers transformers torch scikit-learn nltk
```
### Run the Pipeline
```bash
python app.py --conv conversation.json --ctx context_vectors.json --response_file response.txt
```

## 2. Architecture
The evaluation pipeline consists of five core components:

1. Input Processing
Loads conversation & context JSON
Auto-fixes malformed JSON (trailing commas, BOM, etc.)

2. Embedding & Preprocessing
Converts all texts to strings
Computes embeddings using MiniLM-L6-v2

3. Relevance Scoring
Cosine similarity between:
Response ↔ User query
Response ↔ Context chunks
Generates a combined relevance score

5. Completeness Scoring
Checks if top-k context chunks are reflected in the response
Computes coverage percentage

7. Hallucination Detection
Splits response into sentences
NLI model (BART-MNLI) checks:
Entailment
Contradiction
Unsupported claims
Flags hallucinated sentences

Final Output
Produces a structured report.json with:
Relevance & completeness
Sentence-level hallucination detection
Latency timings
Token & cost estimations


## 3. Why This Design?
**Fast & lightweight:** MiniLM + BART-MNLI run efficiently on CPU.

**Explainable:** Shows which sentences are unsupported.

**Modular:** Any component (embedding, NLI, scoring) can be replaced independently.

**Robust:** Handles messy JSON inputs and mixed data types gracefully.


## 4. Scalability for Millions of Daily Evaluations
The pipeline is optimized for low latency and minimal compute cost:

Pre-computed Embeddings
Context embeddings stored in the vector DB avoid recomputation.

**Batching**

Vector similarity and NLI calls run in batches → higher throughput.

**Tiered Evaluation**

Only uncertain responses trigger heavy NLI checks → large cost reduction.

**CPU-Optimized Models**

No GPU requirement → significantly lowers cloud costs.

**Early Exit Logic**

High similarity → skip NLI
Low similarity → skip deep checks
Reduces unnecessary computation.

**Asynchronous Sampling**

Only 1–5% responses undergo deeper audits → scalable quality monitoring.
