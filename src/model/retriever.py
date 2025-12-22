import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GNNRetriever(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, question_dim):
        super(GNNRetriever, self).__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        
        # Scoring layer: combines node embedding and question embedding
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + question_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, question_embed):
        # x: [num_nodes, node_dim]
        # question_embed: [batch_size, question_dim]
        
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        
        # Expand question embedding to match nodes
        # Assuming single graph for simplicity, otherwise use batch indices
        num_nodes = x.size(0)
        q_expanded = question_embed.expand(num_nodes, -1)
        
        logits = self.scorer(torch.cat([x, q_expanded], dim=-1))
        return logits.squeeze(-1)

