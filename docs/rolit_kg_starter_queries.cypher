// RoLit-KG MVP starter queries

// 1) Works and their characters
MATCH (w:Work)-[:HAS_CHARACTER]->(c:Character)
RETURN w.title AS work, collect(distinct c.canonical_name)[0..20] AS characters
ORDER BY work;

// 2) Top character interactions (co-occurrence heuristic)
MATCH (c1:Character)-[r:INTERACTS_WITH]->(c2:Character)
RETURN c1.canonical_name AS c1, c2.canonical_name AS c2, count(*) AS times, avg(r.confidence) AS conf
ORDER BY times DESC
LIMIT 20;

// 3) Grounded historical figures: characters based on people
MATCH (c:Character)-[r:BASED_ON]->(p:Person)
RETURN c.canonical_name AS character, p.canonical_name AS person, r.confidence AS confidence
ORDER BY confidence DESC
LIMIT 50;

// 4) Entities located in locations
MATCH (e:Entity)-[r:LOCATED_IN]->(l:Location)
RETURN e.canonical_name AS entity, l.canonical_name AS location, count(*) AS mentions
ORDER BY mentions DESC
LIMIT 50;

