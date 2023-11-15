import torch
import torch.nn.functional as F
import torch.nn as nn


def style_loss(model, pred, style):

    _, pred_skips = model.module.encoder(pred)

    with torch.no_grad():

        btm, style_skips = model.module.encoder(style)

    crit = nn.MSELoss(reduction='sum')
    loss = 0
    for i, (pred_skip, style_skip) in enumerate(zip(pred_skips, style_skips)):

        loss += crit(F.relu(pred_skip).mean(), F.relu(style_skip).mean())
        loss += crit(F.relu(pred_skip).std(), F.relu(style_skip).std())

    return loss

