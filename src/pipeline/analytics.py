"""
Graph analytics for RoLit-KG.

Features:
- Centrality metrics (PageRank, Betweenness, Degree)
- Community detection
- Narrative pattern mining
- Character network analysis
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import json


@dataclass
class GraphMetrics:
    """Computed graph metrics."""
    node_count: int
    edge_count: int
    avg_degree: float
    density: float
    
    top_nodes_by_degree: List[Tuple[str, int]]
    top_nodes_by_pagerank: List[Tuple[str, float]]
    
    communities: List[List[str]]
    community_count: int
    
    narrative_patterns: List[Dict[str, any]]


class GraphAnalyzer:
    """Analyzes the knowledge graph structure."""
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: List[Tuple[str, str, str]] = []  # (source, predicate, target)
        self.adj_list: Dict[str, List[str]] = defaultdict(list)
        self.node_types: Dict[str, str] = {}
        self.node_names: Dict[str, str] = {}
    
    def build_from_artifacts(
        self,
        entities: List,
        relations: List,
    ) -> None:
        """Build graph from entities and relations."""
        # Add nodes
        for e in entities:
            entity_id = e.entity_id if hasattr(e, 'entity_id') else e['entity_id']
            self.nodes.add(entity_id)
            self.node_types[entity_id] = e.entity_type if hasattr(e, 'entity_type') else e['entity_type']
            self.node_names[entity_id] = e.canonical_name if hasattr(e, 'canonical_name') else e['canonical_name']
        
        # Add edges
        for r in relations:
            source_id = r.source_entity_id if hasattr(r, 'source_entity_id') else r['source_entity_id']
            target_id = r.target_entity_id if hasattr(r, 'target_entity_id') else r['target_entity_id']
            predicate = r.predicate if hasattr(r, 'predicate') else r['predicate']
            
            self.edges.append((source_id, predicate, target_id))
            self.adj_list[source_id].append(target_id)
            self.adj_list[target_id].append(source_id)  # Treat as undirected for some metrics
    
    def compute_metrics(self) -> GraphMetrics:
        """Compute comprehensive graph metrics."""
        # Basic metrics
        node_count = len(self.nodes)
        edge_count = len(self.edges)
        
        total_degree = sum(len(neighbors) for neighbors in self.adj_list.values())
        avg_degree = total_degree / node_count if node_count > 0 else 0.0
        
        max_edges = node_count * (node_count - 1)
        density = edge_count / max_edges if max_edges > 0 else 0.0
        
        # Degree centrality
        degree_centrality = {
            node: len(self.adj_list[node])
            for node in self.nodes
        }
        top_by_degree = sorted(
            [(self.node_names.get(n, n), deg) for n, deg in degree_centrality.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:20]
        
        # PageRank
        pagerank_scores = self._compute_pagerank()
        top_by_pagerank = sorted(
            [(self.node_names.get(n, n), score) for n, score in pagerank_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:20]
        
        # Community detection
        communities = self._detect_communities()
        
        # Narrative patterns
        patterns = self._mine_narrative_patterns()
        
        return GraphMetrics(
            node_count=node_count,
            edge_count=edge_count,
            avg_degree=avg_degree,
            density=density,
            top_nodes_by_degree=top_by_degree,
            top_nodes_by_pagerank=top_by_pagerank,
            communities=communities,
            community_count=len(communities),
            narrative_patterns=patterns,
        )
    
    def _compute_pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """Compute PageRank using power iteration."""
        if not self.nodes:
            return {}
        
        n = len(self.nodes)
        node_list = list(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        # Initialize scores
        scores = [1.0 / n] * n
        
        # Build adjacency structure
        out_degree = [len(self.adj_list[node]) for node in node_list]
        
        for iteration in range(max_iter):
            new_scores = [(1 - damping) / n] * n
            
            for i, node in enumerate(node_list):
                # Distribute score to neighbors
                if out_degree[i] > 0:
                    contrib = damping * scores[i] / out_degree[i]
                    for neighbor in self.adj_list[node]:
                        if neighbor in node_to_idx:
                            j = node_to_idx[neighbor]
                            new_scores[j] += contrib
            
            # Check convergence
            diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            if diff < tol:
                break
            
            scores = new_scores
        
        return {node_list[i]: scores[i] for i in range(n)}
    
    def _detect_communities(self) -> List[List[str]]:
        """Simple community detection using connected components."""
        visited = set()
        communities = []
        
        def dfs(node: str, community: List[str]):
            if node in visited:
                return
            visited.add(node)
            community.append(self.node_names.get(node, node))
            
            for neighbor in self.adj_list.get(node, []):
                dfs(neighbor, community)
        
        for node in self.nodes:
            if node not in visited:
                community = []
                dfs(node, community)
                if len(community) > 1:  # Only keep non-trivial communities
                    communities.append(community)
        
        # Sort by size
        communities.sort(key=len, reverse=True)
        return communities[:10]  # Top 10 communities
    
    def _mine_narrative_patterns(self) -> List[Dict[str, any]]:
        """Mine common narrative patterns (triangles, love triangles, conflicts)."""
        patterns = []
        
        # Find love triangles (A loves B, C loves B)
        love_edges = [(s, t) for s, p, t in self.edges if p == "LOVES"]
        love_targets = defaultdict(list)
        
        for source, target in love_edges:
            love_targets[target].append(source)
        
        for target, lovers in love_targets.items():
            if len(lovers) >= 2:
                patterns.append({
                    "type": "love_triangle",
                    "target": self.node_names.get(target, target),
                    "suitors": [self.node_names.get(l, l) for l in lovers[:3]],
                    "count": len(lovers),
                })
        
        # Find conflict patterns (A kills B, C kills B)
        kill_edges = [(s, t) for s, p, t in self.edges if p == "KILLS"]
        kill_targets = defaultdict(list)
        
        for source, target in kill_edges:
            kill_targets[target].append(source)
        
        for target, killers in kill_targets.items():
            if len(killers) >= 1:
                patterns.append({
                    "type": "death",
                    "victim": self.node_names.get(target, target),
                    "perpetrators": [self.node_names.get(k, k) for k in killers],
                })
        
        # Find family dynasties (chains of DESCENDS_FROM)
        patterns.extend(self._find_family_chains())
        
        return patterns[:50]  # Top 50 patterns
    
    def _find_family_chains(self) -> List[Dict[str, any]]:
        """Find family lineage chains."""
        descent_edges = [(s, t) for s, p, t in self.edges if p == "DESCENDS_FROM"]
        
        # Build parent map
        parent_map = {child: parent for child, parent in descent_edges}
        
        # Find chains
        chains = []
        visited = set()
        
        for node in parent_map.keys():
            if node in visited:
                continue
            
            # Trace back to root
            chain = []
            current = node
            while current and current not in visited:
                chain.append(self.node_names.get(current, current))
                visited.add(current)
                current = parent_map.get(current)
            
            if len(chain) >= 2:
                chains.append({
                    "type": "family_lineage",
                    "lineage": list(reversed(chain)),
                    "depth": len(chain),
                })
        
        return sorted(chains, key=lambda x: x["depth"], reverse=True)[:10]
    
    def export_metrics_report(self, output_path: str) -> None:
        """Export metrics to JSON file."""
        metrics = self.compute_metrics()
        
        report = {
            "summary": {
                "nodes": metrics.node_count,
                "edges": metrics.edge_count,
                "avg_degree": round(metrics.avg_degree, 2),
                "density": round(metrics.density, 4),
                "communities": metrics.community_count,
            },
            "top_nodes": {
                "by_degree": [
                    {"name": name, "degree": deg}
                    for name, deg in metrics.top_nodes_by_degree
                ],
                "by_pagerank": [
                    {"name": name, "score": round(score, 4)}
                    for name, score in metrics.top_nodes_by_pagerank
                ],
            },
            "communities": [
                {"id": i, "size": len(comm), "members": comm[:10]}
                for i, comm in enumerate(metrics.communities)
            ],
            "narrative_patterns": metrics.narrative_patterns,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Graph analytics report saved to: {output_path}")


def compute_graph_analytics(
    entities: List,
    relations: List,
    output_path: Optional[str] = None,
) -> GraphMetrics:
    """
    Compute graph analytics and optionally export report.
    
    Args:
        entities: List of Entity objects or dicts
        relations: List of Relation objects or dicts
        output_path: Optional path to save JSON report
    
    Returns:
        GraphMetrics object with computed metrics
    """
    analyzer = GraphAnalyzer()
    analyzer.build_from_artifacts(entities, relations)
    metrics = analyzer.compute_metrics()
    
    if output_path:
        analyzer.export_metrics_report(output_path)
    
    return metrics
