
import torch


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    valid = (valid >= 0.5)
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        # assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()  # 
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()  # 
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'train_epe': epe.mean().item(),
        'train_1px': (epe < 1).float().mean().item(),
        'train_3px': (epe < 3).float().mean().item()
    }

    return flow_loss, metrics
