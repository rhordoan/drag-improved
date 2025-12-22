import torch
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, gnn_dim, nemotron_dim=2688):
        super(Projector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(gnn_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, nemotron_dim)
        )

    def forward(self, x):
        # x: [num_selected_facts, gnn_dim]
        return self.net(x)

