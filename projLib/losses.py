import torch

def infonce_loss(z1, z2, sim_metric, criterion, tau=1.0):
    """
    This code originates from the following project:
    - https://github.com/ysharma1126/ssl_identifiability
    """
    sim11 = sim_metric(z1.unsqueeze(-2), z1.unsqueeze(-3)) / tau
    sim22 = sim_metric(z2.unsqueeze(-2), z2.unsqueeze(-3)) / tau
    sim12 = sim_metric(z1.unsqueeze(-2), z2.unsqueeze(-3)) / tau
    d = sim12.shape[-1]
    sim11[..., range(d), range(d)] = float('-inf')
    sim22[..., range(d), range(d)] = float('-inf')
    raw_scores1 = torch.cat([sim12, sim11], dim=-1)
    raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
    raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
    targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
    loss_value = criterion(raw_scores, targets)
    return loss_value