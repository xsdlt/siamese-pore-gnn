"""Explicit shared/three-expert/four-expert encoders for the new comparison."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GraphConv, AttentionalAggregation, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch


def topk_mean_pool(x, batch, k=5):
    dense, mask = to_dense_batch(x, batch)
    dense = dense.masked_fill(~mask.unsqueeze(-1), -torch.inf)
    values = dense.topk(min(k, dense.size(1)), dim=1).values
    valid = torch.isfinite(values)
    return values.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)


class PoolEncoder(nn.Module):
    def __init__(self, node_dim, wide_dim, pools, config, normalize_pool):
        super().__init__()
        hidden = config['hidden_dim']
        self.pools = tuple(pools)
        self.top_k = config['top_k']
        self.conv1 = GraphConv(node_dim, hidden, aggr='add')
        if 'Attn' in pools:
            self.attn = AttentionalAggregation(nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, 1)))
        width = hidden * len(pools)
        self.pool_norm = nn.BatchNorm1d(width) if normalize_pool else nn.Identity()
        self.fc = nn.Sequential(nn.Linear(width + wide_dim, 64), nn.BatchNorm1d(64),
                                nn.ReLU(), nn.Dropout(config['dropout']),
                                nn.Linear(64, config['compress_dim']))

    def forward(self, graph, wide):
        x = F.relu(self.conv1(graph.x, graph.edge_index))
        pooled = []
        for op in self.pools:
            if op == 'Mean':
                value = global_mean_pool(x, graph.batch)
            elif op == 'Max':
                value = global_max_pool(x, graph.batch)
            elif op == 'TopK':
                value = topk_mean_pool(x, graph.batch, self.top_k)
            elif op == 'Attn':
                value = self.attn(x, graph.batch)
            else:
                raise ValueError(op)
            pooled.append(value)
        z = self.pool_norm(torch.cat(pooled, dim=1))
        return self.fc(torch.cat([z, wide.reshape(z.size(0), -1)], dim=1))


class SiamesePoreGNN(nn.Module):
    """The two Siamese sides share this entire model, including every expert."""
    def __init__(self, node_dim, wide_dim, config):
        super().__init__()
        self.architecture = config['architecture']
        if self.architecture == 'shared':
            groups = [('Max', 'Mean', 'TopK', 'Attn')]
        elif self.architecture == 'experts3':
            groups = [('TopK',), ('Attn',), ('Mean', 'Max')]
        elif self.architecture == 'experts4':
            groups = [('TopK',), ('Attn',), ('Mean',), ('Max',)]
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')
        self.experts = nn.ModuleList([PoolEncoder(node_dim, wide_dim, g, config,
                                                 self.architecture != 'shared') for g in groups])
        self.embedding_dim = len(groups) * config['compress_dim']
        width = config['comparison_dim']
        self.fusion = nn.Sequential(nn.Linear(self.embedding_dim, width), nn.BatchNorm1d(width),
                                    nn.ReLU(), nn.Dropout(config['dropout']), nn.Linear(width, 1))

    def forward_one(self, graph, wide):
        return torch.cat([e(graph, wide) for e in self.experts], dim=1)

    def forward(self, graph_a, wide_a, graph_b, wide_b):
        return self.fusion(self.forward_one(graph_a, wide_a) - self.forward_one(graph_b, wide_b)).squeeze(1)
