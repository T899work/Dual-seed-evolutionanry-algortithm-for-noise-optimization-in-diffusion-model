import torch
import torch.optim as optim
import torch.nn as nn

def objective(p1, p2, path, f, eps, p):
    dist_1 = torch.unsqueeze(torch.norm(p1 - path[0, :].reshape(p1.shape)), dim=-1)
    dists = torch.norm(path[0:-1, :] - path[1:, :], dim=1)
    dist_n = torch.unsqueeze(torch.norm(p2 - path[-1, :].reshape(p2.shape)), dim=-1)
    dists = torch.cat((dist_1, dists, dist_n), dim=0)
    value_1 = f(torch.norm((path[0] / 2 + p1.T / 2).T, dim=0, keepdim=True))
    f_values = f(torch.norm(path.T[:, :-1] / 2 + path.T[:, 1:] / 2, dim=0, keepdim=True))
    f_values = f(torch.norm(path.T[:, :-1], dim=0, keepdim=True))
    value_n = f(torch.norm((path[-1] / 2 + p2.T / 2).T, dim=0, keepdim=True))
    f_values = torch.cat((value_1, f_values, value_n), dim=1)
    if p:
        line_integral = (dists * f_values).sum()
    else:
        line_integral = f_values.sum()
    dist_violations = torch.relu((dists - eps)*0)
    penalty = torch.sum(dist_violations)
    return line_integral, penalty


def path_between_two_points(a, b, n, p):
    d = a.shape[0]
    x = torch.linspace(0, 1, n, dtype=torch.float).view(n, 1).repeat(1, d).to("cuda")
    x = x * (b.t() - a.t()) + a.t()
    if p == True:
        for i in range(x.shape[0]):
            x[i] = x[i] + torch.randn(x[i].shape).to("cuda") / (100)
    else:
        for i in range(x.shape[0]):
            x[i] = x[i]/ (2 ** 0.5) +(torch.randn_like(x[i]) / (2 ** 0.5))
    x = nn.Parameter(x[1:-1, :]) 
    return x


def norm_aware_interpolation(f, a, b, n, eps, op, eta=0.001, p=False):
    x = path_between_two_points(a, b, n, p)

    optimizer = optim.Adam([x], lr=0.001)
    value = 10000 if op >= 2 else 0
    for i in range(value):
        optimizer.zero_grad()
        line_integral, penalty = objective(a, b, x, f, eps, p)
        loss = -1 * line_integral
        loss.backward()
        optimizer.step()

    return x.detach()


