from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import pymupdf 
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
PDF_FOLDER = Path("sample_pdfs")
CACHE_DIR = Path(".rag_cache")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT = 300  # seconds to wait for one Ollama reply
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MiniLM only reads ~256 tokens, so chunks are kept a bit smaller than
# before (180 words) so the end of a chunk is not cut off when embedded.
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30

TOP_K = 3                 # passages sent to the LLM for normal questions
COMPARE_TOP_K = 4         # passages for comparison questions
COMPARE_MAX_SOURCES = 3   # how many different PDFs a comparison may lead with
MAX_CHUNKS_PER_PAGE = 2

# Cosine similarity (embeddings are normalized, so dot product = cosine).
# These are STARTING values. Tune them with the /debug command.
MIN_SIMILARITY = 0.25     # below this a chunk is ignored completely
WEAK_SIMILARITY = 0.35    # if the BEST chunk is below this, skip the LLM

WEAK_SUPPORT = 0.30       # min word overlap between a sentence and its cited passage

NUM_CTX = 3072            # same value for every Ollama call (avoids model reloads)
NUM_PREDICT = 320
INSUFFICIENT_MARKER = "INSUFFICIENT_EVIDENCE"
MIN_ANSWER_WORDS = 8      # an answer with fewer non-citation words is rejected
RELATIVE_CUTOFF = 0.75    # drop passages scoring below 75% of the best score

CITE_RE = re.compile(r"\[\s*Source\b[^\]]*\]", re.IGNORECASE)

STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "were", "which", "their",
    "there", "these", "those", "also", "into", "than", "then", "such", "when",
    "where", "while", "about", "other", "more", "most", "some", "used", "using",
    "can", "will", "would", "could", "should", "does", "make", "makes", "based",
    "both", "each", "many", "much", "very", "only", "they", "them", "what",
}


# ----------------------------------------------------------------------
# PDF reading and chunking
# ----------------------------------------------------------------------
def extract_pdf_chunks(pdf_path):
    chunks = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            words = text.split()
            step = CHUNK_SIZE - CHUNK_OVERLAP
            chunk_index = 0

            for start in range(0, len(words), step):
                chunk = " ".join(words[start:start + CHUNK_SIZE])

                if chunk:
                    chunks.append({
                        "text": chunk,
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1

                if start + CHUNK_SIZE >= len(words):
                    break

    return chunks


def list_pdfs():
    return sorted(PDF_FOLDER.glob("*.pdf"))


def load_documents(pdf_files=None):
    pdf_files = pdf_files if pdf_files is not None else list_pdfs()

    if not pdf_files:
        raise ValueError(f"No PDFs found in {PDF_FOLDER.resolve()}.")

    if not 5 <= len(pdf_files) <= 10:
        print(f"Note: found {len(pdf_files)} PDFs (5-10 is the usual range).")

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name}...")
        try:
            chunks = extract_pdf_chunks(pdf_path)
            all_chunks.extend(chunks)
            print(f"  Extracted {len(chunks)} chunks")

            if not chunks:
                print(
                       "  Warning: no text found. This PDF may be scanned or "
                       "image-only (OCR is not supported)."
                     )
        except Exception as exc:
            print(f"  Could not read {pdf_path.name}: {exc}")

    if not all_chunks:
        raise ValueError(
            "No readable PDF text found. Scanned PDFs may require OCR."
        )

    return all_chunks


# ----------------------------------------------------------------------
# Embeddings + on-disk cache
# ----------------------------------------------------------------------
def create_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=True,
    )

    return np.asarray(vectors, dtype=np.float32)


def check_truncation(chunks, model):
    """Warn if chunks are longer than the embedding model can read."""
    limit = model.max_seq_length
    try:
        too_long = sum(
            1 for c in chunks
            if len(model.tokenizer.tokenize(c["text"])) > limit
        )
    except Exception:
        return
    print(f"Embedding model reads up to {limit} tokens per chunk.")
    if too_long:
        print(
            f"Warning: {too_long} of {len(chunks)} chunks are longer than "
            f"that. Lower CHUNK_SIZE if retrieval seems to miss details."
        )


def make_signature(pdf_files):
    return {
        "files": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime_ns,
            }
            for p in pdf_files
        ],
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
    }


def build_or_load_index(model):
    """Reuse saved chunks/embeddings if the PDFs and settings are unchanged."""
    pdf_files = list_pdfs()
    signature = make_signature(pdf_files)

    meta_path = CACHE_DIR / "meta.json"
    chunks_path = CACHE_DIR / "chunks.json"
    vectors_path = CACHE_DIR / "vectors.npy"

    if meta_path.exists() and chunks_path.exists() and vectors_path.exists():
        try:
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if saved == signature:
                chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
                vectors = np.load(vectors_path)
                if len(chunks) == len(vectors):
                    print("Loaded chunks and embeddings from cache.")
                    return chunks, vectors
        except Exception as exc:
            print(f"Cache could not be used ({exc}); rebuilding.")

    chunks = load_documents(pdf_files)
    print(f"Total chunks: {len(chunks)}")
    check_truncation(chunks, model)

    print("\nCreating embeddings...")
    vectors = create_embeddings(chunks, model)

    CACHE_DIR.mkdir(exist_ok=True)
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    np.save(vectors_path, vectors)
    meta_path.write_text(json.dumps(signature), encoding="utf-8")
    print("Saved cache for faster startup next time.")

    return chunks, vectors


# ----------------------------------------------------------------------
# Question analysis (which PDF? which page? comparison?)
# ----------------------------------------------------------------------
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[_\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_filters(question, sources):
    """Work out which PDF(s) and page the user is asking about."""
    q_norm = normalize_text(question)

    page_match = re.search(r"\bpages?\s+(\d+)\b", question.lower())
    page = int(page_match.group(1)) if page_match else None

    # Stem-only matches ("Support Vector Machines") need a cue word so a
    # general question like "what is machine learning" is not filtered.
    has_cue = bool(
        re.search(r"\b(pdf|file|document|according to|based on)\b", q_norm)
    ) or page is not None

    named = []
    for source in sources:
        full = normalize_text(source)
        stem = normalize_text(Path(source).stem)
        if full in q_norm or (has_cue and stem and stem in q_norm):
            named.append(source)

    unknown_file = None
    if not named:
        m = re.search(r"(\S+?)\.pdf\b", question, flags=re.IGNORECASE)
        if m:
            unknown_file = m.group(0)

    comparison = bool(
        re.search(
            r"\b(compare|comparison|difference|differences|differ|differs|"
            r"versus|vs|better than|worse than|contrast)\b",
            question.lower(),
        )
    )

    return {
        "sources": named,
        "page": page,
        "unknown_file": unknown_file,
        "comparison": comparison,
    }


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
def pick_chunks(ranked, chunks, k, initial=None):
    """Pick up to k chunks, avoiding near-duplicates from the same page."""
    chosen = list(initial or [])
    page_counts = Counter(
        (chunks[i]["source"], chunks[i]["page"]) for i in chosen
    )

    for i in ranked:
        if len(chosen) >= k:
            break
        if i in chosen:
            continue

        c = chunks[i]
        key = (c["source"], c["page"])

        if page_counts[key] >= MAX_CHUNKS_PER_PAGE:
            continue

        # Adjacent chunks can contain useful additional context.
        # MAX_CHUNKS_PER_PAGE limits repetition from one page.

        chosen.append(i)
        page_counts[key] += 1

    return chosen


def retrieve(question, chunks, vectors, model):
    """Return a dict: status, message, passages, filters.

    status is one of: ok, no_file, no_page, no_match, weak
    """
    sources = sorted({c["source"] for c in chunks})
    filters = parse_filters(question, sources)

    def result(status, message="", passages=None):
        return {
            "status": status,
            "message": message,
            "passages": passages or [],
            "filters": filters,
        }

    # --- Filter problems are reported clearly, never silently ignored ---
    if filters["unknown_file"]:
        return result(
            "no_file",
            f"I couldn't find a PDF matching '{filters['unknown_file']}'. "
            f"Available PDFs: {', '.join(sources)}",
        )

    candidates = list(range(len(chunks)))

    if filters["sources"]:
        candidates = [
            i for i in candidates if chunks[i]["source"] in filters["sources"]
        ]

    if filters["page"] is not None:
        wanted = filters["page"]
        candidates = [i for i in candidates if chunks[i]["page"] == wanted]
        if not candidates:
            where = (
                ", ".join(filters["sources"])
                if filters["sources"] else "any of your PDFs"
            )
            return result(
                "no_page",
                f"I couldn't find readable text on page {wanted} of {where}. "
                f"The page may not exist or may be scanned/empty.",
            )

    query_vector = model.encode(
        [question], normalize_embeddings=True
    )[0]
    scores = vectors @ query_vector

    ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)

    explicit_scope = bool(filters["sources"] or filters["page"] is not None)

    if not explicit_scope:
        # General question: apply the similarity thresholds.
        ranked = [i for i in ranked if scores[i] >= MIN_SIMILARITY]
        if not ranked:
            best = float(np.max(scores))
            return result(
                "no_match",
                f"No passage in your PDFs was relevant enough to this "
                f"question (best similarity {best:.2f}, minimum "
                f"{MIN_SIMILARITY:.2f}). I won't answer from general "
                f"knowledge when you're asking about your PDFs.",
            )

    # Drop passages far weaker than the best match so off-topic text is not
    # sent to the LLM. Skipped for comparisons (a second PDF can legitimately
    # score lower) and when the user named a PDF or page.
    if not explicit_scope and not filters["comparison"]:
        best_score = scores[ranked[0]]
        ranked = [i for i in ranked if scores[i] >= best_score * RELATIVE_CUTOFF]

    k = COMPARE_TOP_K if filters["comparison"] else TOP_K

    initial = []
    if filters["comparison"]:
        # Make sure different PDFs are represented, not just the top one.
        best_per_source = {}
        for i in ranked:
            best_per_source.setdefault(chunks[i]["source"], i)
        initial = sorted(
            best_per_source.values(), key=lambda i: scores[i], reverse=True
        )[:COMPARE_MAX_SOURCES]
        # A forced "best passage per PDF" is unrelated when that PDF has
        # nothing on the topic, so keep it only if it clears WEAK_SIMILARITY.
        # Skipped when the user named a PDF or page.
        if not explicit_scope:
            initial = [i for i in initial if scores[i] >= WEAK_SIMILARITY]

    chosen = pick_chunks(ranked, chunks, k, initial=initial)

    passages = []
    for i in chosen:
        p = dict(chunks[i])
        p["similarity"] = float(scores[i])
        passages.append(p)

    # Only judge "weak" on unfiltered searches; with an explicit PDF/page
    # the user told us where to look, so the LLM gets to decide.
    if not explicit_scope and passages[0]["similarity"] < WEAK_SIMILARITY:
        return result(
            "weak",
            f"The closest passages are only weakly related "
            f"(best similarity {passages[0]['similarity']:.2f}), so I don't "
            f"trust them enough to answer. Try rephrasing, or name the PDF.",
            passages,
        )

    return result("ok", "", passages)


def debug_retrieval(question, chunks, vectors, model, show=10):
    """Print raw scores so thresholds can be tuned from real data."""
    sources = sorted({c["source"] for c in chunks})
    filters = parse_filters(question, sources)
    print(f"\nFilters detected: {filters}")

    query_vector = model.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ query_vector

    print(f"\nTop {show} chunks by cosine similarity (no threshold):")
    for rank, i in enumerate(np.argsort(scores)[::-1][:show], start=1):
        c = chunks[int(i)]
        preview = c["text"][:70].replace("\n", " ")
        print(
            f"{rank:>2}. {scores[i]:.3f} | {c['source']} | page {c['page']} "
            f"| {preview}..."
        )

    res = retrieve(question, chunks, vectors, model)
    print(f"\nRetrieve status: {res['status']}")
    if res["message"]:
        print(res["message"])
    for p in res["passages"]:
        print(
            f"  selected: {p['source']} p{p['page']} "
            f"(sim {p['similarity']:.3f})"
        )
    print(
        f"\nCurrent thresholds: MIN_SIMILARITY={MIN_SIMILARITY}, "
        f"WEAK_SIMILARITY={WEAK_SIMILARITY}, TOP_K={TOP_K}, "
        f"RELATIVE_CUTOFF={RELATIVE_CUTOFF}"
    )


# ----------------------------------------------------------------------
# Ollama helper
# ----------------------------------------------------------------------
class OllamaError(requests.RequestException):
    """Ollama is unreachable, too slow, missing the model, or sent a bad reply."""


def call_ollama(prompt, num_predict):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "num_predict": num_predict,
                    "num_ctx": NUM_CTX,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        text = data["response"]
        if not isinstance(text, str):
            raise KeyError("response")
        timing = {
            "generation_s": data.get("eval_duration", 0) / 1e9,
            "prompt_s": data.get("prompt_eval_duration", 0) / 1e9,
            "truncated": data.get("done_reason") == "length",
        }
    except requests.Timeout:
        raise OllamaError(
            f"Ollama did not answer within {OLLAMA_TIMEOUT} seconds. The "
            f"model may still be loading or the machine may be busy. Try "
            f"again, or raise OLLAMA_TIMEOUT."
        ) from None
    except requests.ConnectionError:
        raise OllamaError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. Start Ollama (open "
            f"the app or run 'ollama serve') and try again."
        ) from None
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else None
        if code == 404:
            raise OllamaError(
                f"Ollama does not have the model '{LLM_MODEL}'. "
                f"Run: ollama pull {LLM_MODEL}"
            ) from None
        raise OllamaError(f"Ollama returned an HTTP error ({code}).") from None
    except (ValueError, KeyError, TypeError, AttributeError):
        raise OllamaError(
            "Ollama returned a reply that could not be read."
        ) from None

    return text.strip(), timing
# ----------------------------------------------------------------------
# Citations: build labels, check them in Python
# ----------------------------------------------------------------------
def make_source_map(passages):
    """Label -> exact filename, page and text of the passage sent to the LLM."""
    source_map = {}
    for n, p in enumerate(passages, start=1):
        source_map[n] = {
            "label": f"[Source {n}: {p['source']}, page {p['page']}]",
            "source": p["source"],
            "page": p["page"],
            "text": p["text"],
        }
    return source_map


def content_words(text):
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 3 and w not in STOPWORDS
    }


def analyze_answer(raw, source_map):
    """Check citations in the model's answer.

    Returns:
      rendered  - answer with every [Source N] replaced by the real label
      cited     - set of valid source numbers actually cited
      invalid   - labels that do not exist in this answer's context
      uncited   - substantive sentences with no citation
      weak      - cited sentences with little word overlap with their passage
    """
    cited, invalid = set(), set()

    def tokenize(match):
        ids = re.findall(r"Source\s*(\d+)", match.group(0), flags=re.IGNORECASE)
        if not ids:
            invalid.add("?")
            return "\u27e6?\u27e7"
        for x in ids:
            if int(x) in source_map:
                cited.add(int(x))
            else:
                invalid.add(x)
        return "\u27e6" + ",".join(ids) + "\u27e7"

    tokenized = CITE_RE.sub(tokenize, raw)

    uncited, weak = [], []
    sentences = re.findall(
        r"[^.!?\n\u27e6]+[.!?]*(?:\s*\u27e6[^\u27e7]*\u27e7)*", tokenized
    )

    for sentence in sentences:
        cites = re.findall(r"\u27e6([^\u27e7]*)\u27e7", sentence)
        body = re.sub(r"\u27e6[^\u27e7]*\u27e7", "", sentence).strip()

        if len(body.split()) < 6 or body.endswith(":"):
            continue

        if not cites:
            uncited.append(body)
            continue

        ids = {
            int(x) for group in cites for x in group.split(",")
            if x.isdigit() and int(x) in source_map
        }
        if not ids:
            continue

        passage_words = set()
        for n in ids:
            passage_words |= content_words(source_map[n]["text"])
        sentence_words = content_words(body)

        if len(sentence_words) >= 3:
            overlap = len(sentence_words & passage_words) / len(sentence_words)
            if overlap < WEAK_SUPPORT:
                weak.append(body)

    def render(match):
        ids, seen = [], set()
        for x in match.group(1).split(","):
            if x.isdigit() and int(x) in source_map and int(x) not in seen:
                ids.append(int(x))
                seen.add(int(x))
        return " ".join(source_map[n]["label"] for n in ids)

    rendered = re.sub(r"\u27e6([^\u27e7]*)\u27e7", render, tokenized)
    rendered = re.sub(r"[ \t]+\n", "\n", rendered).strip()

    return {
        "rendered": rendered,
        "cited": cited,
        "invalid": invalid,
        "uncited": uncited,
        "weak": weak,
        # Words left after removing every citation tag. A reply that is
        # only "[Source 1]" has almost none and is not a real answer.
        "body_words": len(CITE_RE.sub("", raw).split()),
    }


# ----------------------------------------------------------------------
# Answer generation
# ----------------------------------------------------------------------
def build_prompt(question, context, valid_labels, feedback=""):
    return f"""You answer questions using ONLY the PDF excerpts below.

Rules:
1. Use only facts stated in the excerpts. Do not use outside knowledge, and do not guess how a cut-off excerpt would continue.
2. Put a label like [Source 1] at the end of EVERY sentence, not just at the end of the answer. Valid labels: {valid_labels}. Never invent labels. In a numbered list, end each item with its label.
3. For comparisons, describe each side separately with its own citations. If the excerpts cover only one side, say which side is missing.
4. If the excerpts do not contain the answer, reply with exactly: {INSUFFICIENT_MARKER}
5. If the excerpts answer only part of the question, answer that part and say what is missing.
6. Use simple, student-friendly language. Be concise.
7. If the question asks what a page or PDF says, summarize the main points of the excerpts in 2-4 full sentences.
8. Never reply with only a label. Always write sentences, each followed by its label, e.g. "Support vectors are the points closest to the boundary [Source 1]. Only some training points are used [Source 1]."
{feedback}
Excerpts:
{context}

Question: {question}

Reminder: every sentence must end with its [Source N] label.

Answer:
"""


def answer_question(question, retrieval):
    """Return a dict with status, answer/message, cited sources and notes."""
    passages = retrieval["passages"]
    out = {
        "status": retrieval["status"],
        "answer": None,
        "message": retrieval["message"],
        "cited_sources": [],
        "notes": [],
        "passages": passages,
        "timings": {},
    }

    # Retrieval problems (bad file, bad page, nothing relevant) never
    # reach the LLM, so nothing can be invented.
    if retrieval["status"] != "ok":
        return out

    source_map = make_source_map(passages)
    context = "\n\n".join(
        f"{info['label']}\n{info['text']}" for info in source_map.values()
    )
    valid_labels = ", ".join(f"[Source {n}]" for n in source_map)

    feedback = ""
    report = None

    for attempt in (1,2):
        raw, timing = call_ollama(
            build_prompt(question, context, valid_labels, feedback),
            NUM_PREDICT,
        )
        out["timings"] = timing

        if raw.strip().upper().startswith(INSUFFICIENT_MARKER):
            out["status"] = "insufficient_evidence"
            out["message"] = (
                "The retrieved PDF passages don't contain enough information "
                "to answer this, so I won't answer from general knowledge. "
                "If you expected it to be covered, try naming the PDF or "
                "rephrasing the question."
            )
            return out

        report = analyze_answer(raw, source_map)

        problems = []
        if report["invalid"]:
            problems.append(
                f"Your answer used labels that do not exist "
                f"({', '.join(sorted(report['invalid']))}). "
                f"Use only {valid_labels}."
            )
        if report["body_words"] < MIN_ANSWER_WORDS:
            problems.append(
                "Your answer contained only citations and no explanation. "
                "Write full sentences that explain the excerpt, each ending "
                "with its [Source N] label."
            )
        elif not report["cited"] and not report["invalid"]:
            problems.append("Your answer had no [Source N] citations.")

        if report["uncited"]:
            problems.append(
                "Every substantive sentence must have its own citation. "
                "Use the source that directly supports that sentence."
            )

        if report["weak"]:
            problems.append(
                "Some citations do not match their sentences well enough. "
                "Check the passages and cite the correct source for each claim."
            )

        if not problems:
            break

        if attempt == 1:
            feedback = (
                "\nCORRECTION: " + " ".join(problems)
                + " Rewrite the answer and cite correctly.\n"
            )

    # Still no real text after one retry -> say so instead of showing a
    # bare citation as if it were an answer.
    if report["body_words"] < MIN_ANSWER_WORDS:
        out["status"] = "empty_answer"
        out["message"] = (
            "The model returned a citation but no actual explanation, so "
            "there is no answer to show. The retrieved passage is listed "
            "below. Try rephrasing, e.g. 'Summarize page 11 of Support "
            "Vector Machines.pdf'."
        )
        return out

    # Still broken after one retry -> clear limitation, no fake citations.
    if report["invalid"] or not report["cited"]:
        out["status"] = "citation_failed"
        out["message"] = (
            "I couldn't produce an answer with valid citations, so I'm not "
            "showing it as a verified answer. The closest passages are "
            "listed below; please check them directly or rephrase the question."
        )
        return out

    out["answer"] = report["rendered"]
    out["cited_sources"] = [
        {
            "label": source_map[n]["label"],
            "source": source_map[n]["source"],
            "page": source_map[n]["page"],
        }
        for n in sorted(report["cited"])
    ]
    if report["uncited"]:
        out["status"] = "answered_with_warnings"
        out["notes"].append(
            f"{len(report['uncited'])} sentence(s) have no citation of "
            f"their own, so they can't be traced to a specific page. "
            f"Check them against the cited pages."
        )
    if report["weak"]:
        out["status"] = "answered_with_warnings"
        for s in report["weak"]:
            out["notes"].append(
                f"Weak match between this sentence and its cited "
                f"passage: \"{s[:100]}\""
            )
    if out["timings"].get("truncated"):
        out["status"] = "answered_with_warnings"
        out["notes"].append(
            "The answer hit the length limit and may be cut off."
        )

    return out


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------
def summarize_pdfs(chunks, only_sources=None):
    """Summarize PDFs. Returns a list of dicts (also used by the UI later)."""
    by_pdf = defaultdict(list)
    for c in chunks:
        by_pdf[c["source"]].append(c)

    results = []

    for filename, pdf_chunks in by_pdf.items():
        if only_sources and filename not in only_sources:
            continue

        pdf_chunks = sorted(
            pdf_chunks, key=lambda c: (c["page"], c["chunk_index"])
        )

        text, pages_used = "", []
        for c in pdf_chunks:
            if len(text) >= 3500:
                break
            text += c["text"] + "\n"
            if c["page"] not in pages_used:
                pages_used.append(c["page"])
        text = text[:3500]

        prompt = f"""
Describe this PDF in 2-3 concise lines.
Mention its main topic and key concepts.
Use only the information in the text.
Do not invent information.

PDF text:
{text}

Summary:
"""
        summary, _ = call_ollama(prompt, 100)
        results.append({
            "source": filename,
            "summary": summary,
            "pages_used": pages_used,
        })

    return results


def print_summaries(summaries):
    for s in summaries:
        pages = s["pages_used"]
        page_text = (
            f"pages {pages[0]}-{pages[-1]}" if len(pages) > 1
            else f"page {pages[0]}"
        )
        print(f"\n{s['source']}")
        print(s["summary"])
        print(f"(Based on the opening text of this PDF: {page_text})")


# ----------------------------------------------------------------------
# Printing results
# ----------------------------------------------------------------------
def print_result(result):
    status = result["status"]

    if result["answer"]:
        print("\nAnswer:\n")
        print(result["answer"])

        if result["cited_sources"]:
            print("\nSources cited in the answer:")
            for s in result["cited_sources"]:
                print(f"  {s['label']}")

        for note in result["notes"]:
            print(f"\nNote: {note}")
    else:
        print(f"\n{result['message']}")

        if (
            status in {"weak", "citation_failed", "empty_answer"}
            and result["passages"]
        ):
            print("\nClosest passages (not verified as answers):")
            for p in result["passages"]:
                preview = p["text"][:150].replace("\n", " ")
                print(
                    f"  {p['source']}, page {p['page']} "
                    f"(similarity {p['similarity']:.2f}): {preview}..."
                )

    timings = result.get("timings") or {}
    if timings:
        print(
            f"\nPrompt processing: {timings['prompt_s']:.2f}s | "
            f"Generation: {timings['generation_s']:.2f}s"
        )

    if result["passages"]:
        print("\nRetrieved passages:")
        for i, p in enumerate(result["passages"], start=1):
            print(
                f"  [{i}] {p['source']} | page {p['page']} | "
                f"similarity {p['similarity']:.3f}"
            )

    print("\n" + "-" * 50 + "\n")

SUMMARY_WORDS_RE = re.compile(r"\b(describe|summari[sz]e)", re.IGNORECASE)
ALL_PDFS_RE = re.compile(r"\b(each|every|all)\b", re.IGNORECASE)


def pdf_summary_request(question, filters):
    """Return (is_summary, targets) for whole-PDF summary requests.

    A question that names a page ("summarize page 10 of X.pdf") is NOT a
    whole-PDF summary: it goes through normal page-filtered retrieval.
    targets is None for "all PDFs".
    """
    if filters["page"] is not None:
        return False, None
    q = question.lower()
    if not SUMMARY_WORDS_RE.search(q):
        return False, None
    wants_all = "pdf" in q and bool(ALL_PDFS_RE.search(q))
    if wants_all or filters["sources"]:
        return True, (None if wants_all else filters["sources"])
    return False, None
# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    print("\n=== AI Study Assistant ===")

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nLoading PDFs...")
    try:
        chunks, vectors = build_or_load_index(model)
    except ValueError as exc:
        print(f"\nCannot start: {exc}")
        print(f"Put text-based PDFs in '{PDF_FOLDER}/' and run again.")
        return
    sources = sorted({c["source"] for c in chunks})

    print("\nReady! Ask questions about your PDFs.")
    print("Commands: /files  |  /debug <question>  |  exit")
    print("Tip: name a PDF (e.g. 'in Notes.pdf') or a page ('page 5').\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        if question.lower() == "/files":
            pages = defaultdict(set)
            for c in chunks:
                pages[c["source"]].add(c["page"])
            for name in sources:
                print(f"  {name} ({len(pages[name])} pages with text)")
            print()
            continue

        if question.lower().startswith("/debug"):
            q = question[6:].strip()
            if q:
                debug_retrieval(q, chunks, vectors, model)
            else:
                print("Usage: /debug <your question>\n")
            continue

        try:
            filters = parse_filters(question, sources)
            is_summary, targets = pdf_summary_request(question, filters)

            if is_summary:
                print_summaries(summarize_pdfs(chunks, targets))
                print("\n" + "-" * 50 + "\n")
                continue

            retrieval = retrieve(question, chunks, vectors, model)
            result = answer_question(question, retrieval)
            print_result(result)

        except OllamaError as exc:
            print(f"\nOllama problem: {exc}\n")
        except requests.RequestException as exc:
            print(f"\nNetwork error: {exc}\n")


if __name__ == "__main__":
    main()