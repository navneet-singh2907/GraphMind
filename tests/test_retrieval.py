import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent.orchestrator import RetrievalPlan, ToolCall, Verification, answer_agentic
from src.graph.ontology import DEFAULT_ONTOLOGY, load_ontology
from src.retrieval.keyword_search import KeywordIndex
from src.retrieval.models import Evidence
from src.retrieval.graph_rag import query_graph


class RetrievalTests(unittest.TestCase):
    def test_keyword_search_ranks_exact_terms(self):
        records = [
            {"id": "1", "content": "GraphRAG traverses a knowledge graph.", "source_uri": "a.md"},
            {"id": "2", "content": "Spreadsheets contain rows and columns.", "source_uri": "b.md"},
        ]
        result = KeywordIndex(records).search("knowledge graph")
        self.assertEqual(result[0]["id"], "1")

    def test_default_ontology_has_structural_contract(self):
        self.assertIn("Document", DEFAULT_ONTOLOGY.node_labels)
        self.assertIn("Chunk", DEFAULT_ONTOLOGY.node_labels)
        self.assertIn("PART_OF", DEFAULT_ONTOLOGY.relationship_types)
        self.assertIn("DISCUSSES", DEFAULT_ONTOLOGY.relationship_types)

    def test_custom_ontology_loads_without_python_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ontology.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "legal",
                        "node_labels": {
                            "Document": "Source",
                            "Chunk": "Passage",
                            "Clause": "Contract clause"
                        },
                        "extraction_labels": ["Clause"],
                        "relationship_types": {
                            "PART_OF": "Structural",
                            "DISCUSSES": "Evidence",
                            "REQUIRES": "Requirement"
                        }
                    }
                ),
                encoding="utf-8",
            )
            ontology = load_ontology(path)
            self.assertEqual(ontology.name, "legal")
            self.assertIn("Clause", ontology.extraction_labels)

    @patch("src.retrieval.graph_rag.run_cypher")
    @patch("src.retrieval.graph_rag.generate_cypher")
    def test_graph_query_retries_after_neo4j_rejects_cypher(self, mock_generate, mock_run):
        mock_generate.side_effect = ["MATCH bad RETURN bad", "MATCH good RETURN good"]
        mock_run.side_effect = [ValueError("expected Path but was List<Path>"), [{"good": "result"}]]

        cypher, rows = query_graph("How are the services connected?")

        self.assertEqual(cypher, "MATCH good RETURN good")
        self.assertEqual(rows, [{"good": "result"}])
        self.assertIn("expected Path", mock_generate.call_args_list[1].kwargs["correction"])

    @patch("src.retrieval.graph_rag.run_cypher")
    @patch("src.retrieval.graph_rag.generate_cypher")
    def test_graph_query_broadens_valid_empty_query(self, mock_generate, mock_run):
        mock_generate.side_effect = ["MATCH narrow RETURN narrow", "MATCH broad RETURN broad"]
        mock_run.side_effect = [[], [{"connected": "Retrieval Engine"}]]

        cypher, rows = query_graph("What is connected to Gateway?")

        self.assertEqual(cypher, "MATCH broad RETURN broad")
        self.assertEqual(rows, [{"connected": "Retrieval Engine"}])
        self.assertIn("returned no rows", mock_generate.call_args_list[1].kwargs["correction"])

    @patch("src.agent.orchestrator.synthesize_answer", return_value="Grounded answer [S1]")
    @patch("src.agent.orchestrator.verify_evidence", return_value=Verification(True, 0.9, ""))
    @patch("src.agent.orchestrator.execute_plan")
    @patch("src.agent.orchestrator.plan_question")
    def test_agent_plans_retrieves_verifies_and_answers(
        self, mock_plan, mock_execute, _mock_verify, _mock_synthesize
    ):
        mock_plan.return_value = RetrievalPlan(
            "Use exact and semantic evidence.", [ToolCall("keyword_search", {"query": "GraphMind"})]
        )
        mock_execute.return_value = [
            Evidence(tool="keyword_search", content="GraphMind uses verified retrieval.", source_uri="readme.md")
        ]
        result = answer_agentic("How does GraphMind retrieve?", max_attempts=2)
        self.assertEqual(result["route"], "agentic")
        self.assertEqual(result["attempts"], 1)
        self.assertTrue(result["verification"]["sufficient"])
        self.assertEqual(result["answer"], "Grounded answer [S1]")


if __name__ == "__main__":
    unittest.main()
