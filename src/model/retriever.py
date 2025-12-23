"""
D-RAG GNN-based Retriever

Implementation based on D-RAG paper:
- Section 4.1: Core retriever architecture
- Appendix B: ReaRev-based architecture details
- Appendix G: Training hyperparameters

Architecture follows ReaRev (Mavromatis and Karypis, 2022):
1. Instruction Module: Sentence-BERT encoder for question -> instructions
2. Graph Reasoning Module: Message passing with instruction-conditioned nodes
3. Instruction Update Module: Iterative refinement based on node representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from transformers import AutoModel, AutoTokenizer


class InstructionModule(nn.Module):
    """
    Encodes natural language questions into instruction embeddings using Sentence-BERT.
    These instructions guide the graph reasoning process.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 instruction_dim=384, freeze_encoder=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.instruction_dim = instruction_dim
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, questions):
        """
        Args:
            questions: List of question strings or pre-tokenized input
        Returns:
            instructions: [batch_size, instruction_dim]
        """
        if isinstance(questions, list):
            encoded = self.tokenizer(
                questions,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.encoder.device)
        else:
            encoded = questions
        
        outputs = self.encoder(**encoded)
        instructions = self.mean_pooling(outputs, encoded['attention_mask'])
        return instructions
    
    def to(self, device):
        self.encoder = self.encoder.to(device)
        return super().to(device)


class RelationEncoder(nn.Module):
    """
    Encodes KG relations using LM encoder + MLP projection.
    Shares the LM encoder with the Instruction Module.
    """
    def __init__(self, lm_encoder, lm_dim=384, relation_dim=256):
        super().__init__()
        self.lm_encoder = lm_encoder  # Shared Sentence-BERT
        self.projection = nn.Sequential(
            nn.Linear(lm_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim)
        )
        self.relation_dim = relation_dim
    
    def forward(self, relation_texts, attention_mask=None):
        """
        Args:
            relation_texts: Relation text embeddings from LM [num_relations, lm_dim]
        Returns:
            relation_embeds: [num_relations, relation_dim]
        """
        return self.projection(relation_texts)


class InstructionConditionedConv(MessagePassing):
    """
    Graph convolution that conditions message passing on current instructions.
    Part of the Graph Reasoning Module.
    """
    def __init__(self, node_dim, edge_dim, instruction_dim, hidden_dim, heads=4):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads
        
        # Node transformation conditioned on instruction
        self.node_transform = nn.Linear(node_dim + instruction_dim, hidden_dim)
        
        # Attention mechanism
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature transformation
        self.edge_transform = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, edge_index, edge_attr, instruction):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            instruction: Current instruction [batch_size, instruction_dim]
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        
        # Expand instruction to all nodes (assuming single graph per batch for now)
        if instruction.dim() == 2:
            instruction = instruction.squeeze(0) if instruction.size(0) == 1 else instruction.mean(0)
        instruction_expanded = instruction.unsqueeze(0).expand(num_nodes, -1)
        
        # Condition nodes on instruction
        x_conditioned = torch.cat([x, instruction_expanded], dim=-1)
        x_transformed = self.node_transform(x_conditioned)
        
        # Message passing with attention
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr)
        
        # Residual connection and layer norm
        out = self.layer_norm(x_transformed + self.dropout(out))
        
        return out
    
    def message(self, x_i, x_j, edge_attr, index, size_i):
        """
        Compute attention-weighted messages.
        x_i: target node features [num_edges, hidden_dim]
        x_j: source node features [num_edges, hidden_dim]
        """
        # Multi-head attention
        q = self.query(x_i).view(-1, self.heads, self.head_dim)
        k = self.key(x_j).view(-1, self.heads, self.head_dim)
        v = self.value(x_j).view(-1, self.heads, self.head_dim)
        
        # Attention scores
        attn = (q * k).sum(dim=-1) / (self.head_dim ** 0.5)
        
        # Add edge features to attention if available
        if edge_attr is not None and self.edge_transform is not None:
            edge_features = self.edge_transform(edge_attr)
            edge_features = edge_features.view(-1, self.heads, self.head_dim)
            attn = attn + (q * edge_features).sum(dim=-1) / (self.head_dim ** 0.5)
        
        # Softmax over neighbors
        attn = softmax(attn, index)
        attn = self.dropout(attn)
        
        # Weighted sum of values
        out = (attn.unsqueeze(-1) * v).view(-1, self.hidden_dim)
        
        return out
    
    def update(self, aggr_out):
        return self.out_proj(aggr_out)


class InstructionUpdateModule(nn.Module):
    """
    Updates instructions based on node representations and predicted distributions.
    Allows the model to adapt its search strategy during reasoning.
    """
    def __init__(self, instruction_dim, node_dim, hidden_dim=256):
        super().__init__()
        self.instruction_dim = instruction_dim
        
        # Attention over nodes to gather relevant information
        self.node_attention = nn.Sequential(
            nn.Linear(node_dim + instruction_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # GRU cell for instruction update
        self.gru = nn.GRUCell(node_dim, instruction_dim)
        
        self.layer_norm = nn.LayerNorm(instruction_dim)
    
    def forward(self, instruction, node_embeds, node_scores=None):
        """
        Args:
            instruction: Current instruction [batch_size, instruction_dim]
            node_embeds: Node representations [num_nodes, node_dim]
            node_scores: Optional terminal node predictions [num_nodes]
        Returns:
            Updated instruction [batch_size, instruction_dim]
        """
        num_nodes = node_embeds.size(0)
        
        # Expand instruction for attention computation
        if instruction.dim() == 1:
            instruction = instruction.unsqueeze(0)
        inst_expanded = instruction.expand(num_nodes, -1)
        
        # Compute attention weights over nodes
        attn_input = torch.cat([node_embeds, inst_expanded], dim=-1)
        attn_weights = self.node_attention(attn_input)
        
        # If we have node scores, incorporate them
        if node_scores is not None:
            attn_weights = attn_weights + node_scores.unsqueeze(-1)
        
        attn_weights = F.softmax(attn_weights, dim=0)
        
        # Aggregate node information
        context = (attn_weights * node_embeds).sum(dim=0, keepdim=True)
        
        # Update instruction via GRU
        new_instruction = self.gru(context, instruction)
        new_instruction = self.layer_norm(new_instruction)
        
        return new_instruction


class FactScorer(nn.Module):
    """
    Scores facts (triples) for selection.
    F_i = [h_i || r_i || t_i] -> p(τ_i) = σ(W * F_i + b)
    """
    def __init__(self, entity_dim, relation_dim):
        super().__init__()
        # Fact representation: [head || relation || tail]
        fact_dim = 2 * entity_dim + relation_dim
        
        # Linear layer + sigmoid for Bernoulli probability
        self.scorer = nn.Linear(fact_dim, 1)
    
    def forward(self, head_embeds, relation_embeds, tail_embeds):
        """
        Args:
            head_embeds: [num_facts, entity_dim]
            relation_embeds: [num_facts, relation_dim]
            tail_embeds: [num_facts, entity_dim]
        Returns:
            selection_probs: [num_facts] - Bernoulli parameters
        """
        # Construct fact representations: F_i = [h_i || r_i || t_i]
        fact_repr = torch.cat([head_embeds, relation_embeds, tail_embeds], dim=-1)
        
        # p(τ_i) = σ(W * F_i + b)
        logits = self.scorer(fact_repr)
        probs = torch.sigmoid(logits.squeeze(-1))
        
        return probs


class DRAGRetriever(nn.Module):
    """
    Complete D-RAG Retriever based on ReaRev architecture.
    
    Training:
        - Loss: L = ρ * L_BCE + (1-ρ) * L_Rank, where ρ = 0.7
        - Pre-training: 10 epochs on heuristic subgraphs
        - Joint training: 18 epochs with generator
        - Optimizer: AdamW, lr=5e-5, weight_decay=0.001
        - Batch size: 16
    """
    def __init__(
        self,
        node_dim: int = 256,
        edge_dim: int = 256,
        hidden_dim: int = 256,
        instruction_dim: int = 384,
        relation_dim: int = 256,
        num_reasoning_steps: int = 3,
        num_heads: int = 4,
        lm_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_lm: bool = False,
        rho: float = 0.7,  # BCE vs Rank loss weight
    ):
        super().__init__()
        
        self.num_reasoning_steps = num_reasoning_steps
        self.hidden_dim = hidden_dim
        self.rho = rho
        
        # 1. Instruction Module (Sentence-BERT)
        self.instruction_module = InstructionModule(
            model_name=lm_model_name,
            instruction_dim=instruction_dim,
            freeze_encoder=freeze_lm
        )
        
        # 2. Relation Encoder (shares LM with instruction module)
        self.relation_encoder = RelationEncoder(
            lm_encoder=self.instruction_module,
            lm_dim=instruction_dim,
            relation_dim=relation_dim
        )
        
        # 3. Node initialization from instruction
        self.node_init = nn.Linear(node_dim, hidden_dim)
        
        # 4. Graph Reasoning Module - multiple layers for iterative reasoning
        self.reasoning_layers = nn.ModuleList([
            InstructionConditionedConv(
                node_dim=hidden_dim if i > 0 else hidden_dim,
                edge_dim=edge_dim,
                instruction_dim=instruction_dim,
                hidden_dim=hidden_dim,
                heads=num_heads
            )
            for i in range(num_reasoning_steps)
        ])
        
        # 5. Instruction Update Module
        self.instruction_update = InstructionUpdateModule(
            instruction_dim=instruction_dim,
            node_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # 6. Fact Scorer
        self.fact_scorer = FactScorer(
            entity_dim=hidden_dim,
            relation_dim=relation_dim
        )
        
        # Output dimension for downstream projector
        self.output_dim = 2 * hidden_dim + relation_dim  # [h || r || t]
    
    def encode_relations(self, relation_texts):
        """
        Encode relation texts using the shared LM encoder.
        """
        with torch.no_grad():
            instruction_embeds = self.instruction_module(relation_texts)
        relation_embeds = self.relation_encoder(instruction_embeds)
        return relation_embeds
    
    def forward(
        self,
        node_features,      # [num_nodes, node_dim]
        edge_index,         # [2, num_edges]
        edge_attr,          # [num_edges, edge_dim]
        edge_relations,     # [num_edges, relation_dim] - pre-encoded relation embeddings
        questions,          # List of question strings or tensor
        fact_indices=None,  # Optional: [num_facts, 3] - (head_idx, edge_idx, tail_idx) for each fact
    ):
        """
        Forward pass of the D-RAG retriever.
        
        Returns:
            fact_probs: [num_facts] - Selection probabilities for each fact
            node_embeds: [num_nodes, hidden_dim] - Final node embeddings
            fact_embeds: [num_facts, fact_dim] - Fact representations for projector
        """
        device = node_features.device
        
        # 1. Get initial instructions from questions
        if isinstance(questions, list):
            instruction = self.instruction_module(questions)
        else:
            instruction = questions  # Pre-encoded
        
        # 2. Initialize node representations
        node_embeds = self.node_init(node_features)
        
        # 3. Iterative Graph Reasoning with Instruction Updates
        for layer_idx, conv_layer in enumerate(self.reasoning_layers):
            # Graph reasoning step
            node_embeds = conv_layer(
                node_embeds, 
                edge_index, 
                edge_attr, 
                instruction
            )
            
            # Update instructions based on current node states (except last layer)
            if layer_idx < self.num_reasoning_steps - 1:
                instruction = self.instruction_update(instruction, node_embeds)
        
        # 4. Construct fact representations and compute scores
        if fact_indices is not None:
            # fact_indices: [num_facts, 3] where each row is (head_idx, edge_idx, tail_idx)
            head_idx = fact_indices[:, 0]
            edge_idx = fact_indices[:, 1]
            tail_idx = fact_indices[:, 2]
            
            head_embeds = node_embeds[head_idx]
            tail_embeds = node_embeds[tail_idx]
            relation_embeds = edge_relations[edge_idx]
        else:
            # Default: treat each edge as a fact
            head_embeds = node_embeds[edge_index[0]]
            tail_embeds = node_embeds[edge_index[1]]
            relation_embeds = edge_relations
        
        # 5. Score facts: p(τ_i) = σ(W * [h || r || t] + b)
        fact_probs = self.fact_scorer(head_embeds, relation_embeds, tail_embeds)
        
        # 6. Construct fact embeddings for projector
        fact_embeds = torch.cat([head_embeds, relation_embeds, tail_embeds], dim=-1)
        
        return fact_probs, node_embeds, fact_embeds
    
    def compute_loss(self, fact_probs, labels, margin=1.0):
        """
        Compute combined BCE + Ranking loss.
        
        L = ρ * L_BCE + (1-ρ) * L_Rank
        
        Args:
            fact_probs: [num_facts] - Predicted selection probabilities
            labels: [num_facts] - Binary labels (1 for relevant facts, 0 otherwise)
            margin: Margin for ranking loss
        
        Returns:
            total_loss, bce_loss, rank_loss
        """
        # BCE Loss
        bce_loss = F.binary_cross_entropy(fact_probs, labels.float())
        
        # Ranking Loss - maximize margin between positive and negative facts
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_probs = fact_probs[pos_mask]
            neg_probs = fact_probs[neg_mask]
            
            # Pairwise ranking: for each positive, compare with all negatives
            # Efficient approximation: compare with mean/max negative
            pos_expanded = pos_probs.unsqueeze(1)  # [num_pos, 1]
            neg_expanded = neg_probs.unsqueeze(0)  # [1, num_neg]
            
            # Hinge loss: max(0, margin - (pos - neg))
            rank_loss = F.relu(margin - (pos_expanded - neg_expanded))
            rank_loss = rank_loss.mean()
        else:
            rank_loss = torch.tensor(0.0, device=fact_probs.device)
        
        # Combined loss
        total_loss = self.rho * bce_loss + (1 - self.rho) * rank_loss
        
        return total_loss, bce_loss, rank_loss
    
    def select_facts(self, fact_probs, fact_embeds, k=None, threshold=0.5):
        """
        Select facts based on predicted probabilities.
        
        Args:
            fact_probs: [num_facts] - Selection probabilities
            fact_embeds: [num_facts, fact_dim] - Fact embeddings
            k: If provided, select top-k facts; otherwise use threshold
            threshold: Selection threshold if k is None
        
        Returns:
            selected_embeds: [num_selected, fact_dim]
            selected_indices: [num_selected]
            selected_probs: [num_selected]
        """
        if k is not None:
            # Top-k selection
            k = min(k, fact_probs.size(0))
            selected_probs, selected_indices = torch.topk(fact_probs, k)
        else:
            # Threshold-based selection
            selected_indices = (fact_probs >= threshold).nonzero(as_tuple=True)[0]
            selected_probs = fact_probs[selected_indices]
        
        selected_embeds = fact_embeds[selected_indices]
        
        return selected_embeds, selected_indices, selected_probs
    
    def get_optimizer(self, lr=5e-5, weight_decay=0.001):
        """
        Get AdamW optimizer with paper-specified hyperparameters.
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )


# Backward compatibility alias
GNNRetriever = DRAGRetriever


def create_retriever(
    node_dim=256,
    edge_dim=256,
    hidden_dim=256,
    num_reasoning_steps=3,
    **kwargs
):
    """Factory function to create a D-RAG retriever with default settings."""
    return DRAGRetriever(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_reasoning_steps=num_reasoning_steps,
        **kwargs
    )
