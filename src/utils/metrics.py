"""
D-RAG Evaluation Metrics (based on Section 5.1 and Appendix F)

Answer Generation Metrics:
- Hits@1: Does any correct answer appear in the generated response?
- F1 Score: Token-level F1 between generated and ground truth

Retrieval Metrics:
- Recall: Proportion of relevant facts successfully retrieved
- Precision: Proportion of retrieved facts that are relevant
- Retriever F1: Harmonic mean of precision and recall

Subsets:
- Full Dataset: All examples
- Retrieved Subset: Examples where at least one relevant fact was retrieved
"""

import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Optional


def normalize_answer(s: str) -> str:
    """
    Normalize answer for comparison.
    Lowercases, removes punctuation, articles, and extra whitespace.
    """
    if s is None:
        return ""
    
    s = s.lower()
    
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    
    # Remove extra whitespace
    s = ' '.join(s.split())
    
    return s.strip()


def get_tokens(s: str) -> List[str]:
    """
    Get normalized tokens from a string.

    Note: Some model outputs can come back without spaces (e.g. "DavidDavidDavid" or "MissMiss..."),
    which would otherwise be treated as a single token and collapse token-F1 to 0. We therefore use a
    slightly more robust tokenization:
    - Prefer whitespace tokenization when present
    - Otherwise, fall back to extracting word-like chunks and basic CamelCase-style segments
    """
    norm = normalize_answer(s)
    if not norm:
        return []

    # Standard path (most predictions/GTs): whitespace-delimited.
    toks = norm.split()
    if len(toks) > 1:
        return toks

    # Fallback: extract alphanumeric word chunks.
    # This helps for outputs where spaces were removed.
    chunks = re.findall(r"[a-z0-9]+", norm)
    if len(chunks) > 1:
        return chunks

    # Last-resort: try to split concatenated capitalized entities from the *original* string.
    # Example: "DavidDavidDavid" -> ["david", "david", "david"]
    raw = s or ""
    camel = re.findall(r"[A-Z][a-z]+|[A-Z]+(?![a-z])|[0-9]+", raw)
    if camel:
        return [normalize_answer(t) for t in camel if normalize_answer(t)]

    return toks


# ============================================================================
# Answer Generation Metrics
# ============================================================================

def compute_hits_at_1(prediction: str, ground_truths: List[str]) -> float:
    """
    Hits@1: Check if ANY ground truth answer appears in the prediction.
    
    Args:
        prediction: Generated answer string
        ground_truths: List of acceptable ground truth answers
    
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    pred_normalized = normalize_answer(prediction)
    
    for gt in ground_truths:
        gt_normalized = normalize_answer(gt)
        if gt_normalized and gt_normalized in pred_normalized:
            return 1.0
    
    return 0.0


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """
    Exact Match: Check if prediction exactly matches any ground truth.
    
    Args:
        prediction: Generated answer string
        ground_truths: List of acceptable ground truth answers
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_normalized = normalize_answer(prediction)
    
    for gt in ground_truths:
        if normalize_answer(gt) == pred_normalized:
            return 1.0
    
    return 0.0


def compute_token_f1(prediction: str, ground_truths: List[str]) -> float:
    """
    Token-level F1 score between prediction and best matching ground truth.
    
    Args:
        prediction: Generated answer string
        ground_truths: List of acceptable ground truth answers
    
    Returns:
        Best F1 score across all ground truths
    """
    pred_tokens = get_tokens(prediction)
    
    if not pred_tokens:
        return 0.0
    
    best_f1 = 0.0
    
    for gt in ground_truths:
        gt_tokens = get_tokens(gt)
        
        if not gt_tokens:
            continue
        
        # Count common tokens
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            continue
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        best_f1 = max(best_f1, f1)
    
    return best_f1


# ============================================================================
# Retrieval Metrics
# ============================================================================

def compute_retrieval_metrics(
    fact_probs: List[float],
    labels: List[int],
    threshold: float = 0.5,
    top_k: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute retrieval precision, recall, and F1.
    
    Args:
        fact_probs: Predicted probabilities for each fact
        labels: Binary labels (1 = relevant, 0 = not relevant)
        threshold: Probability threshold for selection (if top_k not used)
        top_k: If provided, select top-k facts instead of thresholding
    
    Returns:
        Dict with 'precision', 'recall', 'f1', 'num_retrieved', 'num_relevant'
    """
    import torch
    
    if isinstance(fact_probs, torch.Tensor):
        fact_probs = fact_probs.detach().cpu().tolist()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    
    n = len(fact_probs)
    
    # Determine which facts are "retrieved"
    if top_k is not None:
        # Top-k selection
        k = min(top_k, n)
        indexed_probs = list(enumerate(fact_probs))
        indexed_probs.sort(key=lambda x: x[1], reverse=True)
        retrieved_indices = set(idx for idx, _ in indexed_probs[:k])
        retrieved = [1 if i in retrieved_indices else 0 for i in range(n)]
    else:
        # Threshold-based selection
        retrieved = [1 if p >= threshold else 0 for p in fact_probs]
    
    # Count
    num_retrieved = sum(retrieved)
    num_relevant = sum(labels)
    num_relevant_retrieved = sum(r * l for r, l in zip(retrieved, labels))
    
    # Compute metrics
    precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0.0
    recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_retrieved': num_retrieved,
        'num_relevant': num_relevant,
        'num_relevant_retrieved': num_relevant_retrieved,
        'any_relevant_retrieved': num_relevant_retrieved > 0
    }


# ============================================================================
# Batch Aggregation
# ============================================================================

class MetricsAccumulator:
    """
    Accumulates metrics across batches for final reporting.
    Tracks both full dataset and "retrieved subset" metrics.
    """
    
    def __init__(self, ret_loss_weight: float = 0.1):
        self.ret_loss_weight = ret_loss_weight
        self.reset()
    
    def reset(self):
        # Answer generation metrics
        self.hits_at_1_sum = 0.0
        self.em_sum = 0.0
        self.f1_sum = 0.0
        self.gen_count = 0
        
        # Retrieval metrics
        self.ret_precision_sum = 0.0
        self.ret_recall_sum = 0.0
        self.ret_f1_sum = 0.0
        self.ret_count = 0
        
        # Retrieved subset (only examples where any relevant fact was retrieved)
        self.hits_at_1_retrieved_sum = 0.0
        self.em_retrieved_sum = 0.0
        self.f1_retrieved_sum = 0.0
        self.retrieved_count = 0
        
        # Loss tracking
        self.gen_loss_sum = 0.0
        self.ret_loss_sum = 0.0
        self.loss_count = 0
    
    def add_generation_result(
        self,
        prediction: str,
        ground_truths: List[str],
        any_relevant_retrieved: bool
    ):
        """Add a single generation result."""
        # Ensure ground_truths is a list
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        
        hits = compute_hits_at_1(prediction, ground_truths)
        em = compute_exact_match(prediction, ground_truths)
        f1 = compute_token_f1(prediction, ground_truths)
        
        # Full dataset
        self.hits_at_1_sum += hits
        self.em_sum += em
        self.f1_sum += f1
        self.gen_count += 1
        
        # Retrieved subset
        if any_relevant_retrieved:
            self.hits_at_1_retrieved_sum += hits
            self.em_retrieved_sum += em
            self.f1_retrieved_sum += f1
            self.retrieved_count += 1
    
    def add_retrieval_result(self, metrics: Dict[str, float]):
        """Add retrieval metrics for a single example."""
        self.ret_precision_sum += metrics['precision']
        self.ret_recall_sum += metrics['recall']
        self.ret_f1_sum += metrics['f1']
        self.ret_count += 1
    
    def add_loss(self, gen_loss: float, ret_loss: float):
        """Add loss values."""
        self.gen_loss_sum += gen_loss
        self.ret_loss_sum += ret_loss
        self.loss_count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all aggregated metrics."""
        metrics = {}
        
        # Generation metrics (full dataset)
        if self.gen_count > 0:
            metrics['hits@1'] = self.hits_at_1_sum / self.gen_count
            metrics['em'] = self.em_sum / self.gen_count
            metrics['gen_f1'] = self.f1_sum / self.gen_count
        
        # Generation metrics (retrieved subset)
        if self.retrieved_count > 0:
            metrics['hits@1_retrieved'] = self.hits_at_1_retrieved_sum / self.retrieved_count
            metrics['em_retrieved'] = self.em_retrieved_sum / self.retrieved_count
            metrics['gen_f1_retrieved'] = self.f1_retrieved_sum / self.retrieved_count
            metrics['retrieved_ratio'] = self.retrieved_count / self.gen_count if self.gen_count > 0 else 0.0
        
        # Retrieval metrics
        if self.ret_count > 0:
            metrics['ret_precision'] = self.ret_precision_sum / self.ret_count
            metrics['ret_recall'] = self.ret_recall_sum / self.ret_count
            metrics['ret_f1'] = self.ret_f1_sum / self.ret_count
        
        # Loss metrics
        if self.loss_count > 0:
            metrics['gen_loss'] = self.gen_loss_sum / self.loss_count
            metrics['ret_loss'] = self.ret_loss_sum / self.loss_count
            metrics['combined_loss'] = metrics['gen_loss'] + self.ret_loss_weight * metrics['ret_loss']
        
        return metrics
    
    def format_report(self, prefix: str = "") -> str:
        """Format metrics as a readable string."""
        m = self.get_metrics()
        
        lines = []
        
        # Loss
        if 'gen_loss' in m:
            lines.append(f"{prefix}Loss: {m.get('combined_loss', 0):.4f} "
                        f"(Gen: {m['gen_loss']:.4f}, Ret: {m['ret_loss']:.4f})")
        
        # Retrieval metrics
        if 'ret_precision' in m:
            lines.append(f"{prefix}Retrieval: P={m['ret_precision']:.4f}, "
                        f"R={m['ret_recall']:.4f}, F1={m['ret_f1']:.4f}")
        
        # Generation metrics (full)
        if 'hits@1' in m:
            lines.append(f"{prefix}Generation (Full): Hits@1={m['hits@1']:.4f}, "
                        f"EM={m['em']:.4f}, F1={m['gen_f1']:.4f}")
        
        # Generation metrics (retrieved subset)
        if 'hits@1_retrieved' in m:
            ratio = m.get('retrieved_ratio', 0) * 100
            lines.append(f"{prefix}Generation (Retrieved {ratio:.1f}%): "
                        f"Hits@1={m['hits@1_retrieved']:.4f}, "
                        f"EM={m['em_retrieved']:.4f}, F1={m['gen_f1_retrieved']:.4f}")
        
        return '\n'.join(lines)

