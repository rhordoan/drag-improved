"""
Differentiable Fact Sampler for D-RAG

Implementation based on D-RAG paper:
- Section 4.3.2: Independent Binary Gumbel-Softmax (Equations 7 & 8)
- Appendix A: Straight-Through Estimator (STE) details

Key differences from standard Gumbel-Softmax:
1. Independent binary decisions per fact (not categorical over all facts)
2. STE: Forward pass uses discrete {0,1}, backward pass uses soft gradients
3. Allows variable-sized subgraph selection (0 to N facts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableFactSampler(nn.Module):
    """
    Implements Independent Binary Gumbel-Softmax with Straight-Through Estimator.
    
    Each fact gets an independent binary selection decision:
    - Forward: Discrete {0, 1} (clean input for LLM)
    - Backward: Soft gradients (differentiable)
    
    Equations from paper:
    - Eq. 7: z_soft = softmax((log(p) + Gumbel) / τ)
    - Eq. 8: z = stop_gradient(z_hard - z_soft) + z_soft (STE)
    """
    
    def __init__(self, temp=0.5):
        """
        Args:
            temp: Temperature for Gumbel-Softmax. Lower = more discrete.
                  Paper uses τ=0.5 as default.
        """
        super(DifferentiableFactSampler, self).__init__()
        self.temp = temp
    
    def forward(self, logits):
        """
        Independent binary selection for each fact using Gumbel-Softmax + STE.
        
        Args:
            logits: [num_facts] or [batch_size, num_facts]
                    Raw scores from retriever (before sigmoid)
        
        Returns:
            selection_mask: [num_facts] or [batch_size, num_facts]
                           Binary mask (0 or 1) with gradient flow
        """
        # Handle different input shapes
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # [1, num_facts]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 1. Convert logits to log probabilities for binary classes
        # Class 0: Selected (p), Class 1: Not Selected (1-p)
        # Using logsigmoid for numerical stability
        log_p_select = F.logsigmoid(logits)      # log(σ(x)) = log(p)
        log_p_reject = F.logsigmoid(-logits)     # log(1 - σ(x)) = log(1-p)
        
        # Stack into [batch_size, num_facts, 2]
        # Index 0: probability of selecting
        # Index 1: probability of rejecting
        log_probs = torch.stack([log_p_select, log_p_reject], dim=-1)
        
        # 2. Sample Gumbel noise (Equation 7)
        # Gumbel(0, 1) = -log(-log(U)) where U ~ Uniform(0, 1)
        gumbel_noise = -torch.empty_like(log_probs).exponential_().log()
        
        # 3. Gumbel-Softmax: z_soft = softmax((log_p + gumbel) / τ)
        z_soft = F.softmax((log_probs + gumbel_noise) / self.temp, dim=-1)
        
        # 4. Straight-Through Estimator (Equation 8)
        # Forward pass: discrete one-hot (argmax)
        # Backward pass: soft gradients from z_soft
        
        # Get hard selection (one-hot)
        index = z_soft.max(dim=-1, keepdim=True)[1]  # [batch_size, num_facts, 1]
        z_hard = torch.zeros_like(z_soft).scatter_(-1, index, 1.0)
        
        # STE trick: z = z_hard - z_soft.detach() + z_soft
        # Forward: uses z_hard (discrete)
        # Backward: gradients flow through z_soft
        z_ste = z_hard - z_soft.detach() + z_soft
        
        # 5. Extract selection mask (index 0 = selected)
        selection_mask = z_ste[..., 0]  # [batch_size, num_facts]
        
        if squeeze_output:
            selection_mask = selection_mask.squeeze(0)
        
        return selection_mask
    
    def sample_without_gradient(self, logits):
        """
        Sample facts without gradient (for inference).
        Uses the Gumbel trick but returns hard selections.
        
        Args:
            logits: [num_facts] or [batch_size, num_facts]
        
        Returns:
            selection_mask: [num_facts] or [batch_size, num_facts]
                           Binary mask (0 or 1), no gradients
        """
        with torch.no_grad():
            return self.forward(logits)
    
    def deterministic_select(self, logits, threshold=0.5):
        """
        Deterministic selection based on probabilities (no Gumbel noise).
        Useful for evaluation.
        
        Args:
            logits: [num_facts] or [batch_size, num_facts]
            threshold: Selection threshold (default 0.5)
        
        Returns:
            selection_mask: Binary mask based on threshold
        """
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()


# Backward compatibility
DifferentiableSampler = DifferentiableFactSampler


def gumbel_topk(logits, k, temp=0.5):
    """
    DEPRECATED: Use DifferentiableFactSampler instead.
    
    This function implements categorical Gumbel-Softmax (competition across all facts),
    which differs from the paper's independent binary selection approach.
    
    Kept for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "gumbel_topk uses categorical Gumbel-Softmax, which differs from D-RAG paper's "
        "independent binary selection. Use DifferentiableFactSampler instead.",
        DeprecationWarning
    )
    
    gumbels = -torch.empty_like(logits).exponential_().log()
    scores = (logits + gumbels) / temp
    return F.softmax(scores, dim=-1)


def test_sampler():
    """
    Test function to verify STE gradient flow.
    """
    print("Testing DifferentiableFactSampler...")
    
    # Create sampler
    sampler = DifferentiableFactSampler(temp=0.5)
    
    # Create test logits with gradient tracking
    logits = torch.randn(5, requires_grad=True)  # 5 facts
    
    # Forward pass
    selection = sampler(logits)
    
    print(f"Input logits: {logits.detach()}")
    print(f"Probabilities: {torch.sigmoid(logits).detach()}")
    print(f"Selection (0/1): {selection.detach()}")
    
    # Backward pass
    loss = selection.sum()
    loss.backward()
    
    print(f"Gradients: {logits.grad}")
    
    # Verify gradients exist
    assert logits.grad is not None, "Gradients should flow through STE!"
    assert logits.grad.abs().sum() > 0, "Gradients should be non-zero!"
    
    print("✅ STE working correctly - gradients flow through discrete selection!")
    
    # Test deterministic mode
    print("\nTesting deterministic selection...")
    with torch.no_grad():
        det_selection = sampler.deterministic_select(logits, threshold=0.5)
        print(f"Deterministic selection (threshold=0.5): {det_selection}")


if __name__ == "__main__":
    test_sampler()
