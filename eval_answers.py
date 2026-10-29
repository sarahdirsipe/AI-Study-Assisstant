"""Answer-level evaluation for the AI Study Assistant (Day 21, Part B).

Runs questions from eval/questions.json through the EXISTING doc1 pipeline
(retrieve -> answer_question, or the PDF-summary route that main() uses),
saves everything needed for MANUAL review, and measures latency.

What is automatic here is only a rough first pass:
  - Did the system answer, decline, or fail?
  - Did the answer cite pages that were actually retrieved (doc1 guarantees
    this) and pages your labels mark as relevant?
Faithfulness and completeness are NOT decided by this script. A citation
proves a label exists, not that the sentence is true. You review those by
reading the saved passages (eval/results_answers.json).

Manual rubric (fill eval/answer_review.json):
  relevance         0 = does not address the question, 1 = partly, 2 = yes
  faithfulness      0 = has a claim the cited passage does not support,
                    1 = mostly supported, minor unsupported detail,
                    2 = every claim is supported by its cited passage
  citation_correct  true if each citation supports the sentence it follows
  completeness      0-2, answerable questions only
  abstention_ok     true if answering / declining was the right decision

Usage (project folder, venv active, Ollama running):
    python eval_answers.py
    python eval_answers.py --ids q01,q03,q13
    python eval_answers.py --show-passages --show-raw
    python eval_answers.py --ids q01 --ollama-url http://127.0.0.1:9/api/generate --skip-check
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

import doc1
from eval_retrieval import load_questions, relevant_units


# ----------------------------------------------------------------------
# Ollama call tracking (wraps doc1.call_ollama without changing doc1.py)
# ----------------------------------------------------------------------
class Tracker:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls = []

    def install(self):
        original = doc1.call_ollama
        tracker = self

        def wrapper(prompt, num_predict):
            t0 = time.perf_counter()
            try:
                text, timing = original(prompt, num_predict)
            except Exception:
                tracker.calls.append(
                    {"wall_s": time.perf_counter() - t0, "failed": True}
                )
                raise
            tracker.calls.append({
                "wall_s": time.perf_counter() - t0,
                "prompt_s": timing["prompt_s"],
                "generation_s": timing["generation_s"],
                "truncated": timing["truncated"],
                "raw": text,
                "failed": False,
            })
            return text, timing

        doc1.call_ollama = wrapper


def check_ollama():
    """Return (ok, message). Verifies the server answers and has the model."""
    base = doc1.OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        r = requests.get(base + "/api/tags", timeout=5)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except requests.RequestException as exc:
        return False, f"Cannot reach Ollama at {base}: {exc}"
    except ValueError:
        return False, f"Ollama at {base} returned an invalid reply."
    if not any(n == doc1.LLM_MODEL or n.startswith(doc1.LLM_MODEL) for n in names):
        return False, (
            f"Ollama is running but model '{doc1.LLM_MODEL}' is not installed. "
            f"Installed: {', '.join(names) or 'none'}. "
            f"Run: ollama pull {doc1.LLM_MODEL}"
        )
    return True, f"Ollama OK, model {doc1.LLM_MODEL} available."


# ----------------------------------------------------------------------
# Routing (mirrors main() in doc1.py - update here if main() changes)
# ----------------------------------------------------------------------
def routes_to_summary(question, sources):
    filters = doc1.parse_filters(question, sources)
    return doc1.pdf_summary_request(question, filters)

# ----------------------------------------------------------------------
# Run one question
# ----------------------------------------------------------------------
def run_question(q, chunks, vectors, model, sources, tracker):
    rec = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "answerable": q["answerable"],
        "expected_pages": sorted(
            f"{s} p{p}" for s, p in relevant_units(q)
        ),
        "routed_to": None,
        "retrieval_status": None,
        "retrieval_s": None,
        "answer_s": None,
        "total_s": None,
        "answer_status": None,
        "answer": None,
        "message": None,
        "cited_pages": [],
        "cited_on_labeled_pages": None,
        "notes": [],
        "retrieved": [],
        "attempts": 0,
        "retried": False,
        "ollama_wall_s": None,
        "ollama_prompt_s": None,
        "ollama_generation_s": None,
        "raw_replies": [],
        "summary": None,
        "error": None,
    }

    tracker.reset()
    t_start = time.perf_counter()

    try:
        is_summary, targets = routes_to_summary(q["question"], sources)
        if is_summary:
            rec["routed_to"] = "summary"
            rec["summary"] = doc1.summarize_pdfs(chunks, targets)
        else:
            rec["routed_to"] = "retrieve+answer"

            t1 = time.perf_counter()
            retrieval = doc1.retrieve(q["question"], chunks, vectors, model)
            rec["retrieval_s"] = time.perf_counter() - t1
            rec["retrieval_status"] = retrieval["status"]

            t2 = time.perf_counter()
            out = doc1.answer_question(q["question"], retrieval)
            rec["answer_s"] = time.perf_counter() - t2

            rec["answer_status"] = out["status"]
            rec["answer"] = out["answer"]
            rec["message"] = out["message"]
            rec["notes"] = out["notes"]
            rec["retrieved"] = [
                {
                    "source": p["source"],
                    "page": p["page"],
                    "similarity": round(p["similarity"], 4),
                    "sent_to_llm": retrieval["status"] == "ok",
                    "text": p["text"],
                }
                for p in out["passages"]
            ]

            cited = {(c["source"], c["page"]) for c in out["cited_sources"]}
            rec["cited_pages"] = sorted(f"{s} p{p}" for s, p in cited)
            if q["answerable"] and cited:
                rec["cited_on_labeled_pages"] = len(cited & relevant_units(q))
    except (requests.RequestException, KeyError, ValueError) as exc:
        rec["error"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:300],
        }

    rec["total_s"] = time.perf_counter() - t_start

    calls = tracker.calls
    rec["attempts"] = len(calls)
    rec["retried"] = rec["routed_to"] == "retrieve+answer" and len(calls) > 1
    if calls:
        rec["ollama_wall_s"] = sum(c["wall_s"] for c in calls)
        ok_calls = [c for c in calls if not c.get("failed")]
        rec["ollama_prompt_s"] = sum(c["prompt_s"] for c in ok_calls)
        rec["ollama_generation_s"] = sum(c["generation_s"] for c in ok_calls)
        rec["raw_replies"] = [c["raw"] for c in ok_calls]

    rec["auto_first_pass"] = classify(q, rec)
    return rec


def classify(q, rec):
    """Rough first pass only. Not a faithfulness judgement."""
    if rec["error"]:
        return f"error ({rec['error']['type']})"
    if rec["routed_to"] == "summary":
        return "summary_route (page number ignored; review manually)"
    answered = bool(rec["answer"])
    if q["answerable"]:
        if answered:
            return "answered (review faithfulness)"
        return f"declined_or_failed ({rec['answer_status']})"
    if answered:
        return "possible_hallucination_review"
    return f"correct_abstention ({rec['answer_status']})"


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------
def fmt(x):
    return "-" if x is None else f"{x:.2f}s"


def print_record(rec, show_passages, show_raw):
    print("\n" + "=" * 90)
    print(
        f"[{rec['id']}] {rec['type']} | answerable={rec['answerable']} "
        f"| route={rec['routed_to']}"
    )
    print(f"Q: {rec['question']}")
    if rec["expected_pages"]:
        print(f"Labeled relevant: {', '.join(rec['expected_pages'])}")

    if rec["error"]:
        print(f"ERROR: {rec['error']['type']}: {rec['error']['message']}")

    if rec["routed_to"] == "summary":
        for s in rec["summary"] or []:
            print(f"\nSummary of {s['source']}:\n{s['summary']}")
            print(f"(pages used: {s['pages_used']})")
    else:
        print(
            f"Retrieval status: {rec['retrieval_status']} | "
            f"answer status: {rec['answer_status']}"
        )
        for p in rec["retrieved"]:
            sent = "" if p["sent_to_llm"] else " (NOT sent)"
            print(
                f"  retrieved: {p['source']} p{p['page']} "
                f"(sim {p['similarity']:.3f}){sent}"
            )
            if show_passages:
                print(f"      {p['text'][:600]}")
        if rec["answer"]:
            print(f"\nAnswer:\n{rec['answer']}")
            print(f"\nCited pages: {', '.join(rec['cited_pages']) or 'none'}")
            if rec["cited_on_labeled_pages"] is not None:
                print(
                    f"Cited pages that are labeled relevant: "
                    f"{rec['cited_on_labeled_pages']}/{len(rec['cited_pages'])}"
                )
            for note in rec["notes"]:
                print(f"Note: {note}")
        elif rec["message"]:
            print(f"\nMessage shown to user:\n{rec['message']}")

    if show_raw:
        for i, raw in enumerate(rec["raw_replies"], start=1):
            print(f"\nRaw model reply (attempt {i}):\n{raw}")

    print(
        f"\nTiming: retrieval {fmt(rec['retrieval_s'])} | "
        f"Ollama {fmt(rec['ollama_wall_s'])} "
        f"(prompt {fmt(rec['ollama_prompt_s'])}, "
        f"generation {fmt(rec['ollama_generation_s'])}, "
        f"attempts {rec['attempts']}) | total {fmt(rec['total_s'])}"
    )
    print(f"Auto first pass: {rec['auto_first_pass']}")


def stat_line(label, values):
    values = [v for v in values if v is not None]
    if not values:
        return f"  {label}: n/a"
    return (
        f"  {label}: n={len(values)}  mean {statistics.mean(values):.2f}s  "
        f"median {statistics.median(values):.2f}s  max {max(values):.2f}s"
    )


def print_summary(records):
    print("\n" + "=" * 90)
    print("SUMMARY (automatic first pass - NOT a faithfulness result)")
    print("=" * 90)
    counts = {}
    for r in records:
        key = r["auto_first_pass"].split(" (")[0]
        counts[key] = counts.get(key, 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

    retried = [r["id"] for r in records if r["attempts"] > 1 and r["routed_to"] != "summary"]
    print(f"  questions that needed a retry: {', '.join(retried) or 'none'}")
    errors = [r["id"] for r in records if r["error"]]
    print(f"  questions with errors: {', '.join(errors) or 'none'}")
    flagged = [r["id"] for r in records if r["notes"]]
    print(f"  answers with warning notes: {', '.join(flagged) or 'none'}")

    print("\nLatency")
    print(stat_line("retrieval", [r["retrieval_s"] for r in records]))
    print(stat_line("Ollama (all attempts)", [r["ollama_wall_s"] for r in records]))
    print(stat_line("total per question", [r["total_s"] for r in records if r["attempts"]]))

    print(
        "\nNext: open eval/results_answers.json, read each answer against its "
        "saved passages, and fill eval/answer_review.json."
    )


# ----------------------------------------------------------------------
# Manual review template (never overwrites ratings you already entered)
# ----------------------------------------------------------------------
def ensure_review_template(path, all_questions):
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            sys.exit(f"{path} is not valid JSON: {exc}")

    have = {r["id"] for r in existing}
    added = 0
    for q in all_questions:
        if q["id"] not in have:
            existing.append({
                "id": q["id"],
                "relevance": None,
                "faithfulness": None,
                "citation_correct": None,
                "completeness": None,
                "abstention_ok": None,
                "comments": "",
            })
            added += 1

    if added or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return added


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--out", default="eval/results_answers.json")
    parser.add_argument("--review", default="eval/answer_review.json")
    parser.add_argument("--ids", default="", help="comma-separated ids, e.g. q01,q13")
    parser.add_argument("--ollama-url", default="")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--show-passages", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--timeout", type=int, default=0)
    args = parser.parse_args()

    if args.ollama_url:
        doc1.OLLAMA_URL = args.ollama_url
    if args.timeout:
        doc1.OLLAMA_TIMEOUT = args.timeout

    print("Loading embedding model...")
    model = SentenceTransformer(doc1.EMBEDDING_MODEL)
    chunks, vectors = doc1.build_or_load_index(model)
    sources = sorted({c["source"] for c in chunks})

    all_questions = load_questions(Path(args.questions), chunks)
    questions = all_questions
    if args.ids:
        wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        known = {q["id"] for q in all_questions}
        unknown = [x for x in wanted if x not in known]
        if unknown:
            sys.exit(f"Unknown question id(s): {', '.join(unknown)}")
        questions = [q for q in all_questions if q["id"] in wanted]

    warmup_s = None
    if not args.skip_check:
        ok, msg = check_ollama()
        print(msg)
        if not ok:
            sys.exit(1)
        print("Warming up the LLM (this includes model load time)...")
        t0 = time.perf_counter()
        try:
            doc1.call_ollama("Reply with the word OK.", 5)
            warmup_s = time.perf_counter() - t0
            print(f"LLM warm-up took {warmup_s:.1f}s (cold start, not counted below).")
        except (requests.RequestException, KeyError, ValueError) as exc:
            sys.exit(f"LLM warm-up failed: {type(exc).__name__}: {exc}")

    tracker = Tracker()
    tracker.install()
    model.encode(["warm up"], normalize_embeddings=True)

    records = []
    for q in questions:
        rec = run_question(q, chunks, vectors, model, sources, tracker)
        records.append(rec)
        print_record(rec, args.show_passages, args.show_raw)

    print_summary(records)

    payload = {
        "settings": {
            "llm_model": doc1.LLM_MODEL,
            "embedding_model": doc1.EMBEDDING_MODEL,
            "num_ctx": doc1.NUM_CTX,
            "num_predict": doc1.NUM_PREDICT,
            "top_k": doc1.TOP_K,
            "compare_top_k": doc1.COMPARE_TOP_K,
            "min_similarity": doc1.MIN_SIMILARITY,
            "weak_similarity": doc1.WEAK_SIMILARITY,
            "relative_cutoff": doc1.RELATIVE_CUTOFF,
            "weak_support": doc1.WEAK_SUPPORT,
            "llm_cold_start_s": warmup_s,
        },
        "results": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    added = ensure_review_template(Path(args.review), all_questions)
    print(f"\nSaved results to {out_path}")
    print(
        f"Review template: {args.review} "
        f"({added} new blank entr{'y' if added == 1 else 'ies'} added)"
    )


if __name__ == "__main__":
    main()
