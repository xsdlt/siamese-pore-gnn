"""Portable prediction bundles and a differentiable fixed-graph MPa interface."""
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from .data import GraphSpecimen, PoreDataProcessor
from .model import SiamesePoreGNN
from .training import FittedModel


def save_bundle(fitted, path, identity):
    refs = [dict(sample_id=s.sample_id, strength=float(s.strength), x=s.graph.x.cpu(),
                 pos=s.graph.pos.cpu(), edge_index=s.graph.edge_index.cpu(), wide=s.wide.cpu())
            for s in fitted.references]
    payload = dict(schema_version=2, architecture=fitted.config['model']['architecture'],
                   model_state={k: v.detach().cpu() for k, v in fitted.model.state_dict().items()},
                   config=fitted.config, processor=fitted.processor.state_dict(), references=refs,
                   identity=identity, history=fitted.history,
                   analysis_scope='Fixed graph node features and wide features; graph builder is not differentiable.')
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def load_bundle(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['schema_version'] == 2
    config = payload['config']
    assert payload['architecture'] == config['model']['architecture']
    refs = [GraphSpecimen(r['sample_id'], Data(x=r['x'], pos=r['pos'], edge_index=r['edge_index']),
                          r['wide'], r['strength']) for r in payload['references']]
    proc = PoreDataProcessor.from_state_dict(payload['processor'])
    assert proc.fitted_ids == [r.sample_id for r in refs]
    model = SiamesePoreGNN(refs[0].graph.num_node_features, refs[0].wide.numel(), config['model']).to(device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    return FittedModel(model, proc, refs, config, payload['history'])


def rank_to_strength_tensor(rank_score, strengths, extension=50.):
    strengths = torch.as_tensor(sorted(strengths), dtype=rank_score.dtype, device=rank_score.device)
    values = torch.cat([strengths[:1] - extension, strengths, strengths[-1:] + extension])
    index = rank_score - .5
    # value[0] corresponds to coordinate -1. Linear endpoint extrapolation is retained.
    left = (torch.floor(index).long() + 1).clamp(0, len(values) - 2)
    frac = index - (left.to(index.dtype) - 1)
    return values[left] + frac * (values[left + 1] - values[left])


def strength_from_graph(fitted, graph, wide, device='cpu'):
    """Differentiable full prediction; suitable for feature IG on the frozen graph.

    Call in eval mode. Feature order, scaling and reference set come from the bundle.
    This function does not recompute graph edges or derived features after perturbation.
    """
    model = fitted.model
    if model.training:
        raise ValueError('Analysis/prediction requires model.eval().')
    refs = fitted.references
    reference_graph = Batch.from_data_list([s.graph for s in refs]).to(device)
    reference_wide = torch.stack([s.wide for s in refs]).to(device)
    query_graph = Batch.from_data_list([graph] * len(refs)).to(device)
    query_wide = wide.reshape(1, -1).expand(len(refs), -1).to(device)
    probabilities = torch.sigmoid(model(query_graph, query_wide, reference_graph, reference_wide))
    return rank_to_strength_tensor(probabilities.sum(), [s.strength for s in refs],
                                   fitted.config['prediction']['endpoint_extension_mpa'])


@torch.no_grad()
def predict_strengths(fitted, specimens, device='cpu'):
    fitted.model.to(device).eval()
    predictions = []
    for s in specimens:
        q = fitted.processor.transform(s, fitted.config['graph'])
        predictions.append(float(strength_from_graph(fitted, q.graph, q.wide, device).cpu()))
    return np.asarray(predictions, dtype=float)
