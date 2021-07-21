import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

from loss import Loss


def get_env(env_name="env.npz"):
    env = np.load(env_name)
    coeff, angles, targets = env["coeff"], env["angles"], env["targets"]
    NUM_TASK = len(targets)
    losses = [Loss(t, a, coeff) for t, a in zip(targets, angles)]
    return NUM_TASK, losses


def get_opt(name, param, lr, momentum=0.9):
    if name == "sgd":
        return torch.optim.SGD([param], lr)
    elif name == "sgdm":
        return torch.optim.SGD([param], lr, momentum=momentum)
    elif name == "adam":
        return torch.optim.Adam([param], lr)


def get_canvas(figsize=(16, 16), plot_3d=False):
    mpl.rcParams.update({"font.size": 32})
    plt.style.use(["seaborn-white"])
    plt.close("all")
    fig = plt.figure(dpi=400, figsize=(16, 16))
    if plot_3d:
        ax = fig.gca(projection="3d")
    else:
        ax = fig.gca()
    plt.tight_layout()
    return fig, ax


def draw_loss_surface(fig, ax, losses):
    X = np.linspace(-20.0, 30.0, 200)
    Y = np.linspace(-20.0, 30.0, 200)
    XX, YY = np.meshgrid(X, Y)

    Zs = [loss.get_contour(XX, YY) for loss in losses]

    for Z in Zs:
        cs = ax.contour(X, Y, Z, levels=np.linspace(0, 200, 100), alpha=0.3)
        # fig.colorbar(cs)


def draw_dist_contour(fig, ax, dist, step, grid_size, colobar=False):
    X = np.linspace(-5.0, 15.0, grid_size)
    Y = np.linspace(-5.0, 15.0, grid_size)
    X, Y = np.meshgrid(X, Y)
    cs = ax.contour(X, Y, dist[:, :, step - 1], levels=100, cmap="RdBu")
    if colobar:
        fig.colorbar(cs)


def draw_meta_loss_contour(fig, ax, meta_losses, step, grid_size, colobar=False):
    X = np.linspace(-5.0, 15.0, grid_size)
    Y = np.linspace(-5.0, 15.0, grid_size)
    X, Y = np.meshgrid(X, Y)
    cs = ax.contour(X, Y, meta_losses[:, :, step - 1], levels=100, cmap="RdBu")
    if colobar:
        fig.colorbar(cs)


def draw_meta_loss_surface(ax, meta_losses, step, grid_size):
    X = np.linspace(-5.0, 15.0, grid_size)
    Y = np.linspace(-5.0, 15.0, grid_size)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, meta_losses[:, :, step])
    ax.set_zlabel("Avg. task loss", labelpad=30)
    ax.set_title(f"Avg. task loss surface after {step} steps")


def draw_meta_path(ax, hist):
    ax.plot(hist[0], hist[1], "ro-", markersize=3)
    ax.plot(hist[0][-1], hist[1][-1], "m*", markersize=10)


def draw_task_path(ax, hists, tidx=None):
    if tidx is not None:
        ax.plot(hists[tidx][0], hists[tidx][1], "g-", markersize=3, alpha=0.3)
        ax.plot(hists[tidx][0][-1], hists[tidx][1][-1], "b*", markersize=10)
    else:
        for hist in hists:
            ax.plot(hist[0], hist[1], "g-", markersize=3, alpha=0.3)
            ax.plot(hist[0][-1], hist[1][-1], "b*", markersize=10)


def draw_rs(fig, ax, meta_losses, grid_size, threshold=0.5, colobar=False):
    num_steps = meta_losses.shape[2]
    required_steps = np.ones((grid_size, grid_size)) * num_steps
    for x in range(grid_size):
        for y in range(grid_size):
            for step in range(num_steps):
                if meta_losses[x, y, step] < threshold:
                    required_steps[x, y] = step + 1
                    break

    X = np.linspace(-5.0, 15.0, grid_size)
    Y = np.linspace(-5.0, 15.0, grid_size)
    X, Y = np.meshgrid(X, Y)

    cs = ax.contour(X, Y, required_steps, levels=15, cmap="RdBu")
    if colobar:
        fig.colorbar(cs)
