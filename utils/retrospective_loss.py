from torch import nn


def retrospective_loss_fn(y, y_p, y_gt, K, scaled, L=1):
    a = nn.L1Loss()(y, y_gt)
    b = nn.L1Loss()(y, y_p.detach())
    c = nn.L1Loss()(y_gt, y_p.detach())
    #loss_val = K * (2*a - b + c)
    gamma = K/(K+L)
    loss_val = (a - gamma*b + gamma*c)
    if scaled:
        loss_val *= (K+L)
    return loss_val