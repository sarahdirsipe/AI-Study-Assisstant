"""Retrieval evaluation for the AI Study Assistant (Day 21, Part A).

Runs every question in eval/questions.json through the EXISTING doc1.retrieve()
and compares what it returns with the labels you wrote by hand.

It measures retrieval only. It never calls Ollama.

Usage (from the project folder, venv active):
    python eval_retrieval.py
    python eval_retrieval.py --verbose
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import doc1

VALID_TYPES = {
    "direct", "multi_passage", "comparison", "specific_page",
    "absent", "ambiguous", "irrelevant_risk",
}
RANK_KS = (1, 3, 5)


# ----------------------------------------------------------------------
# Loading and validating the labels
# ----------------------------------------------------------------------
def load_questions(path, chunks):
    if not path.exists():
        sys.exit(f"Questions file not found: {path}")

    try:
        questions = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"{path} is not valid JSON: {exc}")

    indexed = {(c["source"], c["page"]) for c in chunks}
    sources = {c["source"] for c in chunks}
    problems, seen_ids = [], set()

    for n, q in enumerate(questions, start=1):
        qid = q.get("id", f"#{n}")
        if qid in seen_ids:
            problems.append(f"{qid}: duplicate id")
        seen_ids.add(qid)

        if q.get("type") not in VALID_TYPES:
            problems.append(f"{qid}: unknown type {q.get('type')!r}")
        if not str(q.get("question", "")).strip():
            problems.append(f"{qid}: empty question")
        if not isinstance(q.get("answerable"), bool):
            problems.append(f"{qid}: 'answerable' must be true or false")
            continue

        relevant = q.get("relevant", [])
        if q["answerable"] and not relevant:
            problems.append(f"{qid}: answerable but 'relevant' is empty")
        if not q["answerable"] and relevant:
            problems.append(f"{qid}: not answerable but 'relevant' is filled")

        for r in relevant:
            if r.get("source") not in sources:
                problems.append(f"{qid}: unknown source {r.get('source')!r}")
                continue
            if not r.get("pages"):
                problems.append(f"{qid}: no pages for {r['source']}")
            for p in r.get("pages", []):
                if (r["source"], p) not in indexed:
                    problems.append(
                        f"{qid}: {r['source']} page {p} has no indexed text"
                    )

    if problems:
        print("Label problems found - fix these before trusting any metric:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    return questions


def relevant_units(q):
    return {(r["source"], p) for r in q["relevant"] for p in r["pages"]}


# ----------------------------------------------------------------------
# Raw ranking view (cosine only, no thresholds, no filters)
# ----------------------------------------------------------------------
def ranking_view(q, chunks, vectors, model):
    """Rank every PAGE by its best chunk's cosine score."""
    qv = model.encode([q["question"]], normalize_embeddings=True)[0]
    scores = vectors @ qv

    pages, seen = [], set()
    for i in np.argsort(scores)[::-1]:
        key = (chunks[int(i)]["source"], chunks[int(i)]["page"])
        if key not in seen:
            seen.add(key)
            pages.append((key, float(scores[int(i)])))

    rel = relevant_units(q) if q["answerable"] else set()
    best_relevant = max((s for k, s in pages if k in rel), default=None)
    first_rank = next(
        (r for r, (k, _) in enumerate(pages, start=1) if k in rel), None
    )

    out = {
        "top1_score": pages[0][1],
        "top1_page": list(pages[0][0]),
        "best_relevant_score": best_relevant,
        "first_relevant_rank": first_rank,
    }
    if q["answerable"]:
        for k in RANK_KS:
            top = {key for key, _ in pages[:k]}
            found = len(top & rel)
            out[f"recall@{k}"] = found / len(rel)
            out[f"precision@{k}"] = found / k
        out["rr"] = 1.0 / first_rank if first_rank else 0.0
    return out


# ----------------------------------------------------------------------
# Evaluate one question through the real pipeline
# ----------------------------------------------------------------------
def evaluate_question(q, chunks, vectors, model):
    t0 = time.perf_counter()
    res = doc1.retrieve(q["question"], chunks, vectors, model)
    retrieval_s = time.perf_counter() - t0

    status = res["status"]
    filters = res["filters"]
    scoped = bool(filters["sources"] or filters["page"] is not None)

    # Only status "ok" passages are actually sent to the LLM.
    sent = res["passages"] if status == "ok" else []
    sent_units = [(p["source"], p["page"]) for p in sent]

    rec = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "answerable": q["answerable"],
        "status": status,
        "scoped_by_filter": scoped,
        "comparison_detected": filters["comparison"],
        "retrieval_s": retrieval_s,
        "sent": [
            {
                "source": p["source"],
                "page": p["page"],
                "similarity": round(p["similarity"], 4),
            }
            for p in sent
        ],
        "ranking": ranking_view(q, chunks, vectors, model),
    }

    if q["answerable"]:
        rel = relevant_units(q)
        rel_sources = {s for s, _ in rel}
        found = rel & set(sent_units)
        top1 = sent_units[0] if sent_units else None

        rec["recall"] = len(found) / len(rel)
        rec["precision"] = (
            sum(u in rel for u in sent_units) / len(sent_units)
            if sent_units else 0.0
        )
        rec["hit"] = bool(found)
        rec["source_coverage"] = (
            len({s for s, _ in sent_units} & rel_sources) / len(rel_sources)
        )
        rec["top1_source_ok"] = bool(top1 and top1[0] in rel_sources)
        rec["top1_page_ok"] = bool(top1 and top1 in rel)

        if status != "ok":
            rec["outcome"] = "false_rejection"
            # Did the rejected result at least contain a relevant page?
            rec["rejected_had_relevant"] = bool(
                rel & {(p["source"], p["page"]) for p in res["passages"]}
            )
        else:
            rec["outcome"] = "hit" if found else "miss"
    else:
        rec["outcome"] = (
            "rejected_at_retrieval" if status != "ok" else "passed_to_llm"
        )

    return rec


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def print_per_question(records, verbose):
    print("\n" + "=" * 100)
    print("PER-QUESTION RESULTS (pipeline = what would be sent to the LLM)")
    print("=" * 100)
    print(
        f"{'id':<5} {'type':<16} {'status':<9} {'outcome':<22} "
        f"{'R':>5} {'P':>5} {'rank':>5} {'top1':>6} {'ms':>6}  scoped"
    )
    for r in records:
        rank = r["ranking"]["first_relevant_rank"]
        rec_s = f"{r['recall']:.2f}" if r["answerable"] else "-"
        pre_s = f"{r['precision']:.2f}" if r["answerable"] else "-"
        print(
            f"{r['id']:<5} {r['type']:<16} {r['status']:<9} "
            f"{r['outcome']:<22} {rec_s:>5} {pre_s:>5} "
            f"{(str(rank) if rank else '-'):>5} "
            f"{r['ranking']['top1_score']:>6.3f} "
            f"{r['retrieval_s'] * 1000:>6.0f}  "
            f"{'yes' if r['scoped_by_filter'] else ''}"
        )
        if verbose:
            print(f"      Q: {r['question']}")
            for p in r["sent"]:
                print(
                    f"      sent: {p['source']} p{p['page']} "
                    f"(sim {p['similarity']:.3f})"
                )
            if not r["sent"]:
                print("      sent: nothing")


def print_summary(records):
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    print("\n" + "=" * 100)
    print("SUMMARY - ANSWERABLE QUESTIONS (pipeline output)")
    print("=" * 100)
    print(f"Questions: {len(answerable)}")
    hits = sum(r["outcome"] == "hit" for r in answerable)
    misses = sum(r["outcome"] == "miss" for r in answerable)
    rejects = sum(r["outcome"] == "false_rejection" for r in answerable)
    print(f"  success (>=1 relevant page sent): {hits}/{len(answerable)}")
    print(f"  miss (passages sent, none relevant): {misses}")
    print(f"  false rejection (nothing sent to LLM): {rejects}")
    print(f"  mean Recall (page level):    {mean(r['recall'] for r in answerable):.3f}")
    print(f"  mean Precision (page level): {mean(r['precision'] for r in answerable):.3f}")
    print(f"  top-1 source correct: {sum(r['top1_source_ok'] for r in answerable)}/{len(answerable)}")
    print(f"  top-1 page correct:   {sum(r['top1_page_ok'] for r in answerable)}/{len(answerable)}")
    multi = [r for r in answerable if r["type"] in {"comparison", "multi_passage"}]
    if multi:
        print(
            f"  mean source coverage on comparison/multi_passage "
            f"(n={len(multi)}): {mean(r['source_coverage'] for r in multi):.3f}"
        )

    print("\nBy question type (answerable only): success / total")
    by_type = defaultdict(list)
    for r in answerable:
        by_type[r["type"]].append(r)
    for t, rs in sorted(by_type.items()):
        ok = sum(r["outcome"] == "hit" for r in rs)
        print(f"  {t:<16} {ok}/{len(rs)}   mean recall {mean(r['recall'] for r in rs):.2f}")

    # Ranking metrics exclude questions where a PDF/page filter already
    # decided where to look - ranking them would be meaningless.
    unscoped = [r for r in answerable if not r["scoped_by_filter"]]
    print("\n" + "=" * 100)
    print(f"RAW RANKING METRICS (cosine only, no thresholds; n={len(unscoped)} unfiltered answerable questions)")
    print("=" * 100)
    for k in RANK_KS:
        print(
            f"  Recall@{k}: {mean(r['ranking'][f'recall@{k}'] for r in unscoped):.3f}"
            f"   Precision@{k}: {mean(r['ranking'][f'precision@{k}'] for r in unscoped):.3f}"
        )
    print(f"  MRR: {mean(r['ranking']['rr'] for r in unscoped):.3f}")
    skipped = [r["id"] for r in answerable if r["scoped_by_filter"]]
    if skipped:
        print(f"  Excluded (PDF/page filter decided the scope): {', '.join(skipped)}")

    print("\n" + "=" * 100)
    print("SUMMARY - UNANSWERABLE QUESTIONS (abstention at the retrieval stage)")
    print("=" * 100)
    rej = sum(r["outcome"] == "rejected_at_retrieval" for r in unanswerable)
    print(f"Questions: {len(unanswerable)}")
    print(f"  rejected before the LLM: {rej}/{len(unanswerable)}")
    print(f"  passed to the LLM: {len(unanswerable) - rej}")
    print("  (Passed-to-LLM is not automatically a failure: the LLM may still")
    print("   answer INSUFFICIENT_EVIDENCE. Part B checks that.)")

    print("\n" + "=" * 100)
    print("THRESHOLD DIAGNOSTIC (raw top-1 page score per question)")
    print("=" * 100)
    print("Answerable, unfiltered questions:")
    for r in sorted(unscoped, key=lambda r: r["ranking"]["top1_score"]):
        best_rel = r["ranking"]["best_relevant_score"]
        print(
            f"  {r['id']}  top1={r['ranking']['top1_score']:.3f}  "
            f"best_relevant={best_rel:.3f}  status={r['status']}"
        )
    print("Unanswerable questions:")
    for r in sorted(unanswerable, key=lambda r: -r["ranking"]["top1_score"]):
        print(
            f"  {r['id']}  top1={r['ranking']['top1_score']:.3f}  "
            f"status={r['status']}  ({r['type']})"
        )
    print(
        f"\nCurrent settings: MIN_SIMILARITY={doc1.MIN_SIMILARITY}  "
        f"WEAK_SIMILARITY={doc1.WEAK_SIMILARITY}  "
        f"RELATIVE_CUTOFF={doc1.RELATIVE_CUTOFF}  TOP_K={doc1.TOP_K}  "
        f"COMPARE_TOP_K={doc1.COMPARE_TOP_K}"
    )

    times = [r["retrieval_s"] for r in records]
    print(
        f"\nRetrieval time (encode + search): mean {mean(times) * 1000:.0f} ms, "
        f"max {max(times) * 1000:.0f} ms"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--out", default="eval/results_retrieval.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading embedding model...")
    model = SentenceTransformer(doc1.EMBEDDING_MODEL)
    chunks, vectors = doc1.build_or_load_index(model)

    questions = load_questions(Path(args.questions), chunks)
    print(f"Loaded {len(questions)} labelled questions.")

    # Warm-up so the first timed question does not include one-off start cost.
    model.encode(["warm up"], normalize_embeddings=True)

    records = [evaluate_question(q, chunks, vectors, model) for q in questions]

    print_per_question(records, args.verbose)
    print_summary(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "embedding_model": doc1.EMBEDDING_MODEL,
            "chunk_size": doc1.CHUNK_SIZE,
            "chunk_overlap": doc1.CHUNK_OVERLAP,
            "top_k": doc1.TOP_K,
            "compare_top_k": doc1.COMPARE_TOP_K,
            "min_similarity": doc1.MIN_SIMILARITY,
            "weak_similarity": doc1.WEAK_SIMILARITY,
            "relative_cutoff": doc1.RELATIVE_CUTOFF,
            "max_chunks_per_page": doc1.MAX_CHUNKS_PER_PAGE,
            "num_chunks": len(chunks),
        },
        "results": records,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()