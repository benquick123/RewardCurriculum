from torch import nn


def retrospective_loss_fn(y, y_p, y_gt, K, scaled, L=1, distance=nn.L1Loss):
    a = distance()(y, y_gt)
    b = distance()(y, y_p.detach())
    c = distance()(y_gt, y_p.detach())
    #loss_val = K * (2*a - b + c)
    gamma = K/(K+L)
    loss_val = (a - gamma*b + gamma*c)
    if scaled:
        loss_val *= (K+L)
    return loss_val