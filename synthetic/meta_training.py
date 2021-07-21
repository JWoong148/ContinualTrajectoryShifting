from utils import draw_meta_path, get_canvas, draw_meta_loss_contour
from algorithm import reptile, ours, ours_accurate
import torch
import os
import numpy as np

M = 10
K = 80
# inner_opt = "sgdm"
# meta_opt = "sgd"

meta_losses = np.load("synthetic/avg_task_losses_100.npy")
num_steps = meta_losses.shape[2]
required_steps = np.ones((100, 100)) * num_steps
for i in range(100):
    for j in range(100):
        for step in range(num_steps):
            if meta_losses[i, j, step] < 10:
                required_steps[i, j] = step + 1
                break

X = np.linspace(-5.0, 15.0, 100)
Y = np.linspace(-5.0, 15.0, 100)
X, Y = np.meshgrid(X, Y)

for x in [-5.0, 5.0, 15.0]:
    for y in [-5.0, 5.0, 15.0]:
        for K in [80, 120]:
            M = 240 // K
            for lr in [0.05]:
                for inner_opt in ["adam", "sgd", "sgdm"]:
                    fig_dir = f"figures/0125/x_{x}_y_{y}"
                    os.makedirs(fig_dir, exist_ok=True)
                    reptile(
                        phi=torch.tensor((x, y)),
                        inner_opt=inner_opt,
                        lr=lr,
                        inner_step=K,
                        meta_opt="sgd",
                        meta_lr=0.1,
                        meta_step=M,
                        fig_dir=fig_dir,
                    )

                    h1 = ours(
                        phi=torch.tensor((x, y)),
                        inner_opt=inner_opt,
                        lr=lr,
                        inner_step=K,
                        meta_opt="sgd",
                        meta_lr=0.1,
                        meta_step=M,
                        fig_dir=fig_dir,
                    )

                    fig, ax = get_canvas()
                    ax.set_xlim(-5, 15)
                    ax.set_ylim(-5, 15)
                    ax.plot(h1[0], h1[1], "r-", markersize=3, alpha=0.5)
                    ax.plot(h1[0][-1], h1[1][-1], "k*", markersize=10)
                    ax.plot(h1[0][-1::-K], h1[1][-1::-K], "r.", markersize=5)
                    # ax.plot(h3[0], h3[1], "g-", markersize=3, alpha=0.5)
                    # ax.plot(h3[0][-1], h3[1][-1], "k*", markersize=10)
                    # ax.plot(h3[0][-1::-K], h3[1][-1::-K], "g.", markersize=5)

                    cs = ax.contour(X, Y, required_steps, levels=15, cmap="RdBu")
                    fig.colorbar(cs)

                    fig.savefig(f"{fig_dir}/meta_0.1_{M}steps_inner_{inner_opt}_{lr}_{K}steps.png")

                    # ours_accurate(
                    #     phi=torch.tensor((x, y)),
                    #     inner_opt=inner_opt,
                    #     lr=lr,
                    #     inner_step=K,
                    #     meta_opt=meta_opt,
                    #     meta_lr=0.1,
                    #     meta_step=M,
                    #     fig_dir=fig_dir,
                    # )
