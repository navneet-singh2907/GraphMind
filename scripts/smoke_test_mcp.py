import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"


async def run_smoke(question: str) -> None:
    server = StdioServerParameters(
        command=str(PYTHON),
        args=["-m", "src.mcp_server"],
        cwd=str(ROOT),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            print("TOOLS:", ", ".join(tool_names))
            required = {
                "ask_graphmind", "search_knowledge_base", "search_keywords",
                "search_metadata", "inspect_source", "query_knowledge_graph",
                "run_readonly_cypher", "knowledge_stats",
            }
            missing = required - set(tool_names)
            if missing:
                raise RuntimeError(f"Missing MCP tools: {sorted(missing)}")

            stats = await session.call_tool("knowledge_stats", {})
            stats_text = getattr(stats.content[0], "text", "{}")
            stats_data = json.loads(stats_text)
            print(
                "STATS:",
                json.dumps(
                    {
                        "vector_documents": stats_data.get("vector_documents"),
                        "graph_nodes": stats_data.get("graph_nodes"),
                        "graph_relationships": stats_data.get("graph_relationships"),
                        "documents": stats_data.get("documents", 0),
                        "collections": stats_data.get("collections", []),
                        "source_types": stats_data.get("source_types", []),
                    },
                    indent=2,
                ),
            )

            answer = await session.call_tool("ask_graphmind", {"question": question})
            answer_text = getattr(answer.content[0], "text", "{}")
            if answer.isError:
                print("ERROR:", answer_text)
                raise SystemExit(1)
            try:
                answer_data = json.loads(answer_text)
            except json.JSONDecodeError:
                print("RAW ANSWER:", repr(answer_text))
                raise SystemExit(1)
            print("ROUTE:", answer_data.get("route"))
            print("ANSWER:", answer_data.get("answer", "")[:1000])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Smoke-test the GraphMind MCP server.")
    parser.add_argument(
        "--question",
        default="Which concepts are connected to GraphRAG?",
        help="Question to send to the ask_graphmind MCP tool.",
    )
    args = parser.parse_args()
    asyncio.run(run_smoke(args.question))


if __name__ == "__main__":
    main()
