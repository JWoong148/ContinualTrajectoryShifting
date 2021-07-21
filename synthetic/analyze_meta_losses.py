import numpy as np

from utils import draw_meta_loss_contour, draw_meta_loss_surface, get_canvas

num_steps = 125
grid_size = 100
threshold = 0.5
meta_losses = np.load(f"meta_losses_{grid_size}.npy")
print(meta_losses.shape)

min_loss = [np.min(meta_losses[:, :, i]) for i in range(num_steps)]
min_step = num_steps
for i in range(9, num_steps, 10):
    if min_loss[i] < threshold:
        min_step = i + 1
        break

print(f"Required # of steps to acheive meta loss of {threshold}: {min_step}.")

fig, ax = get_canvas()
ax.plot(np.arange(1, num_steps + 1), min_loss)
ax.set_title("Min avg. task loss")
ax.set_xlabel("# inner steps")
ax.set_ylabel("avg. task loss")
fig.savefig("figures/min_meta_loss.png")

mlmo = []
for i in range(num_steps):
    mn, mx, my = 1e9, -1, -1
    for x in range(grid_size):
        for y in range(grid_size):
            if meta_losses[x, y, i] < mn:
                mn, mx, my = meta_losses[x, y, i], x, y
    mlmo.append(meta_losses[mx, my, min_step])

fig, ax = get_canvas()
ax.plot(np.arange(1, num_steps + 1), mlmo)
ax.set_title(f"Avg. task loss (after {min_step} steps) at meta-optima")
ax.set_xlabel("# inner steps")
ax.set_ylabel("avg. task loss")
fig.savefig("figures/meta_loss_at_meta_optima.png")

for step in [1, 10, 80, 120]:
    fig, ax = get_canvas(plot_3d=True)
    draw_meta_loss_surface(ax, meta_losses, step, grid_size)
    ax.set_title(f"Avg. task loss surface after {step} steps")
    fig.savefig(f"figures/meta_loss_surface_at_{step}.png")

    fig, ax = get_canvas()
    draw_meta_loss_contour(fig, ax, meta_losses, step, grid_size)
    ax.set_title(f"Avg. task loss after {step} steps")
    fig.savefig(f"figures/meta_loss_contour_at_{step}.png")

# Required # of steps to acheive meta loss of 0.5
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

fig, ax = get_canvas()
cs = ax.contour(X, Y, required_steps, levels=15, cmap="RdBu")
ax.set_title("Required # of steps to acheive meta loss of 0.5")
fig.colorbar(cs)
fig.savefig("figures/required_steps.png")
