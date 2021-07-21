import pickle

import numpy as np
import torch

from utils import get_canvas, get_env, get_opt

grid_size = 41
X = np.linspace(-5, 15, grid_size)
Y = np.linspace(-5, 15, grid_size)


NUM_TASK, losses = get_env("env.npz")
chd = {}
par = {}
minima = []
threshold = 0.001


def round_to(x, th):
    return np.floor(x / th + 0.5) * th


def meta_update(phi, losses, inner_step):
    phi = torch.tensor(phi)
    grad = torch.zeros_like(phi)
    for loss in losses:
        theta = phi.clone().detach().requires_grad_(True)
        opt = get_opt("sgdm", theta, 0.05)
        for _ in range(inner_step):
            opt.zero_grad()
            loss(theta).backward()
            opt.step()

        grad += (phi - theta).detach().numpy() / NUM_TASK
    phi = phi - 0.1 * grad
    return phi.numpy()


def back(cur):
    if chd[cur][1] is None:
        return
    for nxt in chd[cur][0]:
        if chd[nxt][1] is None:
            chd[nxt][1] = chd[cur][1] + 1
            par[nxt] = par[cur]
            back(nxt)


cands = [(x, y) for x in X for y in Y]

fig, ax = get_canvas()
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)
ax.tick_params(axis="x", labelsize=50)
ax.tick_params(axis="y", labelsize=50)
ax.set_xticks([-5, 0, 5, 10, 15])
ax.set_yticks([-5, 0, 5, 10, 15])

num_step = 100
tot_hist = {}
fname = f"meta_loss_surface_{num_step}.pdf"

for cand in cands:
    cur = cand
    print(cur, end="\t")
    chd[cur] = [[], None]
    hist = [cur]
    for _ in range(1000):
        nxt = meta_update(cur, losses, num_step)
        nxt = round_to(nxt[0], threshold), round_to(nxt[1], threshold)
        hist.append(nxt)
        if cur == nxt:
            minima.append(cur)
            par[cur] = cur
            chd[cur][1] = 0
            back(cur)
            print(f"Minima: {cur[0]:.3f}, {cur[1]:.3f}   Step: {chd[cand][1]}")
            break
        elif nxt in chd.keys():
            chd[nxt][0].append(cur)
            if chd[nxt][1] is None:
                chd[nxt][1] = chd[cur][1] = 0
                par[nxt] = par[cur] = nxt
            back(nxt)
            print(f"back!   Step: {chd[cand][1]}")
            break
        else:
            chd[nxt] = [[cur], None]
            cur = nxt
    hist = np.asarray(hist)
    tot_hist[cur] = hist
    ax.plot(hist[:, 0], hist[:, 1], "k-", linewidth=5, alpha=0.1, markersize=0)
    fig.savefig(fname)
    with open(f"chd_{num_step}.pkl", "wb") as f:
        pickle.dump(chd, f)

    with open(f"hist_{num_step}.pkl", "wb") as f:
        pickle.dump(tot_hist, f)

    with open(f"par_{num_step}.pkl", "wb") as f:
        pickle.dump(par, f)

for m in minima:
    ax.plot(*m, "ro", markersize=30)
fig.savefig(fname)
