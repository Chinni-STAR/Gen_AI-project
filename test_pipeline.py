"""Unit & Integration Tests for Resume RAG Streamlit Pipeline."""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import unittest
from document_parser import extract_text
from rag_pipeline import (
    index_resume,
    resume_exists,
    retrieve_relevant_chunks,
    retrieve_relevant_chunks_with_scores,
    get_rag_diagnostics,
    _validate_and_normalize_screener_schema,
    _parse_json_response,
)


class TestResumeRAGPipeline(unittest.TestCase):

    def test_document_parser_valid_txt(self):
        sample_txt = "John Doe\nSoftware Engineer with 5 years experience in Python, FastAPI, and Docker."
        extracted = extract_text("resume.txt", sample_txt.encode("utf-8"))
        self.assertIn("John Doe", extracted)
        self.assertIn("FastAPI", extracted)

    def test_document_parser_unsupported_format(self):
        with self.assertRaises(ValueError) as ctx:
            extract_text("image.png", b"fake_png_data")
        self.assertIn("Unsupported file format", str(ctx.exception))

    def test_document_parser_empty_file(self):
        with self.assertRaises(ValueError) as ctx:
            extract_text("empty.pdf", b"")
        self.assertIn("empty", str(ctx.exception))

    def test_rag_indexing_and_retrieval(self):
        sample_resume = (
            "Alice Smith\n"
            "Backend Systems Architect\n"
            "Technical Skills: Python, Django, PostgreSQL, Redis, Kubernetes, AWS Lambda, Docker.\n"
            "Experience: Built microservices handling 10,000 requests per second. Reduced API latency by 40%."
        )

        resume_id, num_chunks, chunks = index_resume(sample_resume)
        self.assertTrue(resume_exists(resume_id))
        self.assertGreaterEqual(num_chunks, 1)

        # Test semantic retrieval
        query = "Python Django PostgreSQL experience"
        retrieved = retrieve_relevant_chunks(resume_id, query, k=2)
        self.assertGreaterEqual(len(retrieved), 1)
        self.assertIn("Python", retrieved[0])

        # Test RAG diagnostics trace
        diagnostics = get_rag_diagnostics(resume_id, query)
        self.assertEqual(diagnostics["resume_id"], resume_id)
        self.assertEqual(len(diagnostics["sample_vector_slice"]), 6)
        self.assertGreaterEqual(len(diagnostics["retrieved_results"]), 1)

    def test_schema_normalization_enforces_exact_fields(self):
        raw_llm_json = {
            "overall_match_score": 88,
            "summary": "Candidate is highly qualified.",
            "strengths": ["Python", "FastAPI"],
            "gaps": ["Kubernetes"],
            "suggestions": [
                "Quantify experience with cloud systems.",
            ],
        }

        normalized = _validate_and_normalize_screener_schema(raw_llm_json)

        # Check Fit Score
        self.assertEqual(normalized["fit_score"], 88)
        self.assertEqual(normalized["matched_skills"], ["Python", "FastAPI"])
        self.assertEqual(normalized["missing_skills"], ["Kubernetes"])

        # Check suggestions array size is EXACTLY 3
        self.assertEqual(len(normalized["suggestions"]), 3)
        self.assertTrue(isinstance(normalized["suggestions"][0], str))
        self.assertTrue(isinstance(normalized["suggestions"][1], str))
        self.assertTrue(isinstance(normalized["suggestions"][2], str))

    def test_json_parser_robustness(self):
        markdown_json = """```json
        {
            "fit_score": 92,
            "summary": "Excellent fit.",
            "matched_skills": ["Python"],
            "missing_skills": [],
            "suggestions": ["S1", "S2", "S3"]
        }
        ```"""
        parsed = _parse_json_response(markdown_json)
        self.assertEqual(parsed["fit_score"], 92)


if __name__ == "__main__":
    unittest.main()
