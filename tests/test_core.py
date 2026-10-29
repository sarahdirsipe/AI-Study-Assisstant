"""Unit tests for the AI Study Assistant.

These tests need NO Ollama, NO embedding model download and NO sample PDFs.
Model replies and embeddings are faked. They check the pipeline's logic,
not the quality of real answers (that is what eval_retrieval.py and
eval_answers.py are for).

Run from the project folder:
    python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import doc1  # noqa: E402

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None

SOURCES = ["Clustering.pdf", "Decision Trees.pdf", "Support Vector Machines.pdf"]
TIMING = {"generation_s": 1.0, "prompt_s": 1.0, "truncated": False}
PASSAGE_TEXT = (
    "K-means assigns each sample to its nearest centroid and then updates "
    "the centroids using the mean of the assigned samples."
)
GOOD_SENTENCE = (
    "K-means assigns each sample to its nearest centroid and then updates "
    "the centroids"
)


def passage(source="Clustering.pdf", page=5, text=PASSAGE_TEXT, sim=0.8):
    return {
        "text": text,
        "source": source,
        "page": page,
        "chunk_index": 0,
        "similarity": sim,
    }


def filters_for(question):
    return doc1.parse_filters(question, SOURCES)


# ----------------------------------------------------------------------
# Fake embeddings for retrieve()
# ----------------------------------------------------------------------
class FakeModel:
    """Every question embeds to [1, 0, 0]."""

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        return np.array([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32)


def make_index(specs):
    """specs: list of (source, page, score).

    Each chunk's cosine similarity with the fake question equals `score`.
    """
    chunks, vectors = [], []
    for n, (source, page, score) in enumerate(specs):
        chunks.append({
            "text": f"chunk {n}",
            "source": source,
            "page": page,
            "chunk_index": 0,
        })
        vectors.append([score, float(np.sqrt(1.0 - score ** 2)), 0.0])
    return chunks, np.array(vectors, dtype=np.float32)


# ----------------------------------------------------------------------
class ParseFiltersTests(unittest.TestCase):
    def test_names_pdf_and_page(self):
        f = filters_for(
            "Summarize page 10 of Clustering.pdf, focusing on spectral clustering"
        )
        self.assertEqual(f["sources"], ["Clustering.pdf"])
        self.assertEqual(f["page"], 10)
        self.assertFalse(f["comparison"])

    def test_stem_needs_cue_word(self):
        f = filters_for("What is Clustering?")
        self.assertEqual(f["sources"], [])
        self.assertIsNone(f["page"])

    def test_stem_with_cue_word(self):
        f = filters_for("According to Decision Trees, what is pruning?")
        self.assertEqual(f["sources"], ["Decision Trees.pdf"])

    def test_comparison_detected(self):
        self.assertTrue(
            filters_for("Compare Linear Regression and Decision Trees")["comparison"]
        )
        self.assertFalse(filters_for("What is regularization?")["comparison"])

    def test_unknown_pdf_reported(self):
        f = filters_for("Explain page 2 of Missing.pdf")
        self.assertEqual(f["unknown_file"], "Missing.pdf")
        self.assertEqual(f["sources"], [])

    def test_plain_question_not_filtered(self):
        f = filters_for("How does gradient descent work?")
        self.assertEqual(f["sources"], [])
        self.assertIsNone(f["page"])
        self.assertIsNone(f["unknown_file"])


# ----------------------------------------------------------------------
class PickChunksTests(unittest.TestCase):
    @staticmethod
    def same_page(n):
        return [
            {"text": f"t{i}", "source": "A.pdf", "page": 1, "chunk_index": i}
            for i in range(n)
        ]

    def test_page_cap_respected(self):
        chunks = self.same_page(4)
        chosen = doc1.pick_chunks([0, 1, 2, 3], chunks, k=4)
        self.assertEqual(len(chosen), doc1.MAX_CHUNKS_PER_PAGE)

    def test_adjacent_chunks_are_kept(self):
        # Regression test: the old "skip adjacent chunks" rule dropped the
        # K-means steps chunk for q04. Adjacent chunks must be allowed.
        chunks = self.same_page(2)
        chosen = doc1.pick_chunks([0, 1], chunks, k=3)
        self.assertEqual(chosen, [0, 1][: doc1.MAX_CHUNKS_PER_PAGE])

    def test_k_limit(self):
        chunks = [
            {"text": "t", "source": "A.pdf", "page": p, "chunk_index": 0}
            for p in range(1, 6)
        ]
        chosen = doc1.pick_chunks(list(range(5)), chunks, k=3)
        self.assertEqual(len(chosen), 3)

    def test_initial_picks_come_first_without_duplicates(self):
        chunks = [
            {"text": "t", "source": "A.pdf", "page": p, "chunk_index": 0}
            for p in range(1, 4)
        ]
        chosen = doc1.pick_chunks([0, 1, 2], chunks, k=3, initial=[2])
        self.assertEqual(chosen[0], 2)
        self.assertEqual(len(chosen), len(set(chosen)))


# ----------------------------------------------------------------------
class RetrieveTests(unittest.TestCase):
    def run_retrieve(self, specs, question):
        chunks, vectors = make_index(specs)
        return doc1.retrieve(question, chunks, vectors, FakeModel())

    def test_ok_and_low_scores_dropped(self):
        res = self.run_retrieve(
            [("Alpha.pdf", 1, 0.8), ("Beta.pdf", 1, 0.1)], "What is a centroid?"
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["passages"]), 1)
        self.assertEqual(res["passages"][0]["source"], "Alpha.pdf")

    def test_relative_cutoff_drops_weaker_passages(self):
        rc = doc1.RELATIVE_CUTOFF
        res = self.run_retrieve(
            [
                ("Alpha.pdf", 1, 0.8),
                ("Alpha.pdf", 2, 0.8 * rc - 0.05),
                ("Alpha.pdf", 3, 0.8 * rc + 0.05),
            ],
            "What is a centroid?",
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual({p["page"] for p in res["passages"]}, {1, 3})

    def test_weak_best_match_is_rejected(self):
        mid = (doc1.MIN_SIMILARITY + doc1.WEAK_SIMILARITY) / 2
        res = self.run_retrieve([("Alpha.pdf", 1, mid)], "What is a centroid?")
        self.assertEqual(res["status"], "weak")
        self.assertTrue(res["passages"])

    def test_nothing_relevant_is_no_match(self):
        res = self.run_retrieve([("Alpha.pdf", 1, 0.0)], "What is a centroid?")
        self.assertEqual(res["status"], "no_match")
        self.assertEqual(res["passages"], [])

    def test_unknown_pdf_is_no_file(self):
        res = self.run_retrieve([("Alpha.pdf", 1, 0.8)], "Explain Missing.pdf")
        self.assertEqual(res["status"], "no_file")
        self.assertIn("Missing.pdf", res["message"])

    def test_missing_page_is_no_page(self):
        res = self.run_retrieve(
            [("Alpha.pdf", 1, 0.8)], "Summarize page 99 of Alpha.pdf"
        )
        self.assertEqual(res["status"], "no_page")

    def test_named_pdf_and_page_skip_thresholds(self):
        res = self.run_retrieve(
            [("Alpha.pdf", 1, 0.1)], "Summarize page 1 of Alpha.pdf"
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["passages"]), 1)

    def test_comparison_floor_drops_unrelated_forced_pdf(self):
        strong = [
            ("A.pdf", 1, 0.80), ("A.pdf", 2, 0.70), ("A.pdf", 3, 0.65),
            ("A.pdf", 4, 0.60), ("A.pdf", 5, 0.58), ("A.pdf", 6, 0.55),
        ]
        question = "Compare topic one versus topic two"

        below = doc1.WEAK_SIMILARITY - 0.05
        res = self.run_retrieve(strong + [("B.pdf", 1, below)], question)
        self.assertEqual(res["status"], "ok")
        self.assertNotIn("B.pdf", {p["source"] for p in res["passages"]})

        above = doc1.WEAK_SIMILARITY + 0.05
        res = self.run_retrieve(strong + [("B.pdf", 1, above)], question)
        self.assertIn("B.pdf", {p["source"] for p in res["passages"]})


# ----------------------------------------------------------------------
class AnalyzeAnswerTests(unittest.TestCase):
    def setUp(self):
        self.source_map = doc1.make_source_map([passage()])

    def test_valid_citation_is_rendered(self):
        report = doc1.analyze_answer(f"{GOOD_SENTENCE} [Source 1].", self.source_map)
        self.assertEqual(report["cited"], {1})
        self.assertEqual(report["invalid"], set())
        self.assertIn("[Source 1: Clustering.pdf, page 5]", report["rendered"])

    def test_invalid_label_detected(self):
        report = doc1.analyze_answer(f"{GOOD_SENTENCE} [Source 9].", self.source_map)
        self.assertIn("9", report["invalid"])
        self.assertEqual(report["cited"], set())

    def test_missing_citation_detected(self):
        report = doc1.analyze_answer(
            "The algorithm repeats the two steps until the centroids stop moving.",
            self.source_map,
        )
        self.assertEqual(report["cited"], set())
        self.assertEqual(report["invalid"], set())
        self.assertEqual(len(report["uncited"]), 1)

    def test_label_only_answer_has_no_body(self):
        report = doc1.analyze_answer("[Source 1]", self.source_map)
        self.assertLess(report["body_words"], doc1.MIN_ANSWER_WORDS)

    def test_unrelated_sentence_gets_weak_warning(self):
        report = doc1.analyze_answer(
            "Photosynthesis converts sunlight into chemical energy inside "
            "chloroplasts [Source 1].",
            self.source_map,
        )
        self.assertEqual(len(report["weak"]), 1)

    def test_supported_sentence_is_not_weak(self):
        report = doc1.analyze_answer(
            "K-means assigns each sample to its nearest centroid [Source 1].",
            self.source_map,
        )
        self.assertEqual(report["weak"], [])


# ----------------------------------------------------------------------
class AnswerQuestionTests(unittest.TestCase):
    @staticmethod
    def retrieval(status="ok", passages=None):
        return {
            "status": status,
            "message": "retrieval message",
            "passages": [passage()] if passages is None else passages,
            "filters": {},
        }

    def ask(self, reply, timing=TIMING):
        with mock.patch.object(doc1, "call_ollama", return_value=(reply, timing)):
            return doc1.answer_question("What does K-means do?", self.retrieval())

    def test_supported_answer(self):
        out = self.ask(f"{GOOD_SENTENCE} [Source 1].")
        self.assertTrue(out["answer"])
        self.assertEqual(out["cited_sources"][0]["source"], "Clustering.pdf")
        self.assertEqual(out["cited_sources"][0]["page"], 5)

    def test_model_abstains(self):
        out = self.ask("INSUFFICIENT_EVIDENCE")
        self.assertEqual(out["status"], "insufficient_evidence")
        self.assertIsNone(out["answer"])

    def test_invalid_label_is_never_shown_as_answer(self):
        out = self.ask(f"{GOOD_SENTENCE} [Source 7].")
        self.assertEqual(out["status"], "citation_failed")
        self.assertIsNone(out["answer"])

    def test_answer_without_citations_is_rejected(self):
        out = self.ask(f"{GOOD_SENTENCE}.")
        self.assertEqual(out["status"], "citation_failed")
        self.assertIsNone(out["answer"])

    def test_label_only_reply_is_empty_answer(self):
        out = self.ask("[Source 1]")
        self.assertEqual(out["status"], "empty_answer")
        self.assertIsNone(out["answer"])

    def test_failed_retrieval_never_calls_the_llm(self):
        with mock.patch.object(doc1, "call_ollama") as fake:
            out = doc1.answer_question(
                "anything", self.retrieval(status="no_match", passages=[])
            )
        fake.assert_not_called()
        self.assertEqual(out["status"], "no_match")
        self.assertIsNone(out["answer"])

    def test_ollama_failure_propagates(self):
        with mock.patch.object(
            doc1, "call_ollama", side_effect=doc1.OllamaError("down")
        ):
            with self.assertRaises(doc1.OllamaError):
                doc1.answer_question("anything", self.retrieval())

    def test_truncated_answer_is_flagged(self):
        timing = dict(TIMING, truncated=True)
        out = self.ask(f"{GOOD_SENTENCE} [Source 1].", timing)
        self.assertEqual(out["status"], "answered_with_warnings")
        self.assertTrue(any("length limit" in n for n in out["notes"]))

    def test_uncited_sentence_is_flagged(self):
        reply = (
            f"{GOOD_SENTENCE} [Source 1]. The algorithm repeats these steps "
            f"until the centroids stop moving much."
        )
        out = self.ask(reply)
        self.assertTrue(out["answer"])
        self.assertEqual(out["status"], "answered_with_warnings")
        self.assertTrue(out["notes"])


# ----------------------------------------------------------------------
class CallOllamaTests(unittest.TestCase):
    @staticmethod
    def response(status=200, json_value=None, json_error=None):
        resp = mock.Mock(status_code=status)
        resp.raise_for_status.return_value = None
        if json_error:
            resp.json.side_effect = json_error
        else:
            resp.json.return_value = json_value
        return resp

    def test_success(self):
        resp = self.response(json_value={
            "response": " hello ",
            "eval_duration": 2_000_000_000,
            "prompt_eval_duration": 1_000_000_000,
            "done_reason": "stop",
        })
        with mock.patch("doc1.requests.post", return_value=resp):
            text, timing = doc1.call_ollama("hi", 10)
        self.assertEqual(text, "hello")
        self.assertEqual(timing["generation_s"], 2.0)
        self.assertFalse(timing["truncated"])

    def test_timeout_message(self):
        with mock.patch("doc1.requests.post", side_effect=requests.Timeout()):
            with self.assertRaises(doc1.OllamaError) as ctx:
                doc1.call_ollama("hi", 10)
        self.assertIn("did not answer", str(ctx.exception))

    def test_connection_error_message(self):
        with mock.patch("doc1.requests.post", side_effect=requests.ConnectionError()):
            with self.assertRaises(doc1.OllamaError) as ctx:
                doc1.call_ollama("hi", 10)
        self.assertIn("Cannot connect", str(ctx.exception))

    def test_missing_model_message(self):
        resp = mock.Mock(status_code=404)
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
        with mock.patch("doc1.requests.post", return_value=resp):
            with self.assertRaises(doc1.OllamaError) as ctx:
                doc1.call_ollama("hi", 10)
        self.assertIn("ollama pull", str(ctx.exception))

    def test_invalid_json_message(self):
        resp = self.response(json_error=ValueError("not json"))
        with mock.patch("doc1.requests.post", return_value=resp):
            with self.assertRaises(doc1.OllamaError) as ctx:
                doc1.call_ollama("hi", 10)
        self.assertIn("could not be read", str(ctx.exception))

    def test_reply_without_response_field(self):
        resp = self.response(json_value={"done": True})
        with mock.patch("doc1.requests.post", return_value=resp):
            with self.assertRaises(doc1.OllamaError):
                doc1.call_ollama("hi", 10)


# ----------------------------------------------------------------------
class SummaryRoutingTests(unittest.TestCase):
    def route(self, question):
        return doc1.pdf_summary_request(question, filters_for(question))

    def test_page_question_goes_to_retrieval(self):
        self.assertEqual(
            self.route("Summarize page 10 of Clustering.pdf"), (False, None)
        )

    def test_named_pdf_summary(self):
        self.assertEqual(
            self.route("Summarize Clustering.pdf"), (True, ["Clustering.pdf"])
        )

    def test_all_pdfs_summary(self):
        self.assertEqual(self.route("Describe each PDF in one line"), (True, None))

    def test_all_inside_another_word_is_not_a_trigger(self):
        # "small" contains "all" - this used to trigger an all-PDF summary.
        self.assertEqual(self.route("Describe the small pdf"), (False, None))

    def test_ordinary_question_is_not_a_summary(self):
        self.assertEqual(self.route("What is a centroid?"), (False, None))


# ----------------------------------------------------------------------
@unittest.skipIf(pymupdf is None, "pymupdf is not installed")
class PdfLoadingTests(unittest.TestCase):
    def test_empty_folder_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(doc1, "PDF_FOLDER", Path(tmp)):
                with self.assertRaises(ValueError) as ctx:
                    doc1.load_documents()
        self.assertIn("No PDFs found", str(ctx.exception))

    def test_blank_pdf_gives_no_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(str(path))
            document.close()
            self.assertEqual(doc1.extract_pdf_chunks(path), [])
            with self.assertRaises(ValueError):
                doc1.load_documents([path])

    def test_corrupt_pdf_is_skipped_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pdf"
            path.write_text("this is not a pdf", encoding="utf-8")
            with self.assertRaises(ValueError):
                doc1.load_documents([path])

    def test_text_pdf_is_chunked_with_metadata_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "text.pdf"
            document = pymupdf.open()
            page = document.new_page()
            words = " ".join(f"w{i}" for i in range(300))
            page.insert_textbox(pymupdf.Rect(36, 36, 559, 806), words, fontsize=8)
            document.save(str(path))
            document.close()

            chunks = doc1.extract_pdf_chunks(path)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual([c["chunk_index"] for c in chunks], list(range(len(chunks))))
        for c in chunks:
            self.assertEqual(c["source"], "text.pdf")
            self.assertEqual(c["page"], 1)
            self.assertLessEqual(len(c["text"].split()), doc1.CHUNK_SIZE)
        first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
        step = doc1.CHUNK_SIZE - doc1.CHUNK_OVERLAP
        self.assertEqual(first[step:], second[: doc1.CHUNK_OVERLAP])

    def test_cache_signature_changes_when_pdf_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.pdf"
            path.write_bytes(b"one")
            before = doc1.make_signature([path])
            path.write_bytes(b"one two")
            after = doc1.make_signature([path])
        self.assertNotEqual(before, after)
        self.assertEqual(after["chunk_size"], doc1.CHUNK_SIZE)


if __name__ == "__main__":
    unittest.main()