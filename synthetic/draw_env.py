from utils import draw_loss_surface, get_canvas
import numpy as np
from loss import Loss

NUM_TASK = 8
env = np.load("env.npz")
coeff, angles, targets = env["coeff"], env["angles"], env["targets"]

losses = [Loss(t, a, coeff) for t, a in zip(targets, angles)]


for i, loss in enumerate(losses):
    fig, ax = get_canvas()
    draw_loss_surface(fig, ax, [losses[i]])
    for target in targets:
        ax.plot(*target, "ro", markersize=10)

    fig.savefig(f"losses_{i}.png")
