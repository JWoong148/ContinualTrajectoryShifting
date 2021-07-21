import numpy as np
import ray
import torch

from utils import get_env, get_opt

num_cpus = 20
ray.init(num_cpus=num_cpus)

NUM_TASK, losses = get_env("env.npz")
num_steps = 125
lr = 0.1
grid_size = 100
assert grid_size % num_cpus == 0

chunk_size = grid_size // num_cpus
X = np.linspace(-5.0, 15.0, grid_size)
Y = np.linspace(-5.0, 15.0, grid_size)


def test(theta, lr, num_steps, opt_name="adam"):
    theta_i = [theta.clone().detach().requires_grad_(True) for _ in range(NUM_TASK)]
    opt_i = [get_opt(opt_name, ti, lr) for ti in theta_i]

    avg_loss = np.zeros((num_steps,))
    for i in range(NUM_TASK):
        for step in range(num_steps):
            loss = losses[i](theta_i[i])
            opt_i[i].zero_grad()
            loss.backward()
            opt_i[i].step()

            avg_loss[step] += loss.item() / NUM_TASK
    return avg_loss


@ray.remote
def compute(X, Y):
    avg_task_losses = np.zeros((X.size, Y.size, num_steps))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            avg_task_losses[i, j] = test(torch.tensor((y, x)), lr, num_steps)
    return avg_task_losses


result_ids = []
for i in range(num_cpus):
    result_ids.append(compute.remote(X[i * chunk_size : (i + 1) * chunk_size], Y))

results = ray.get(result_ids)
avg_task_losses = np.concatenate([r[0] for r in results], axis=0)

np.save(f"avg_task_losses_{grid_size}.npy", avg_task_losses)
