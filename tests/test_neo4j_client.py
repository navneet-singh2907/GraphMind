import unittest
from unittest.mock import MagicMock, patch

from src.graph.neo4j_client import get_graph_inventory


class Neo4jClientTests(unittest.TestCase):
    @patch("src.graph.neo4j_client.get_driver")
    def test_graph_inventory_returns_totals_and_label_counts(self, mock_get_driver):
        driver = MagicMock()
        session = MagicMock()
        mock_get_driver.return_value = driver
        driver.session.return_value.__enter__.return_value = session

        summary_result = MagicMock()
        summary_result.single.return_value = {"nodes": 89, "relationships": 231}
        session.run.side_effect = [
            summary_result,
            [
                {"label": "Document", "count": 10},
                {"label": "Chunk", "count": 13},
            ],
        ]

        result = get_graph_inventory()

        self.assertEqual(result["nodes"], 89)
        self.assertEqual(result["relationships"], 231)
        self.assertEqual(result["labels"], {"Document": 10, "Chunk": 13})
        driver.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
