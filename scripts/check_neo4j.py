import sys
sys.path.insert(0, ".")
from src.graph.neo4j_client import get_driver
from src.utils.config import NEO4J_DATABASE

driver = get_driver()
with driver.session(database=NEO4J_DATABASE) as s:
    counts = s.run(
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
    ).data()
    total_nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    total_rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
driver.close()

print("Total nodes :", total_nodes)
print("Total rels  :", total_rels)
print()
for row in counts:
    print(f"  {row['label']}: {row['cnt']}")
