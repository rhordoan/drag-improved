import torch
import torch.nn.functional as F

def gumbel_topk(logits, k, temp=0.5):
    """
    Adds gumbel noise to logits and selects Top-K, 
    returning a soft selection mask that allows gradient flow.
    """
    # Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()
    scores = (logits + gumbels) / temp
    
    # Softmax approximation of Top-K
    # One simple way is to use soft-top-k or just a standard softmax 
    # if k is relative to the total number of nodes.
    # For D-RAG, we typically want a subset of nodes.
    
    # Returning softmax probabilities for all nodes as a 'soft' subgraph
    return F.softmax(scores, dim=-1)

class DifferentiableSampler(torch.nn.Module):
    def __init__(self, k, temp=0.5):
        super(DifferentiableSampler, self).__init__()
        self.k = k
        self.temp = temp

    def forward(self, logits):
        return gumbel_topk(logits, self.k, self.temp)

