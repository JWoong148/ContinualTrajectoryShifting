import matplotlib.pyplot as plt
import numpy as np
from loss import Loss

NUM_TASK = 8

coeff = np.random.rand(4)
angles = [np.random.rand(1) * np.pi * 2 for _ in range(NUM_TASK)]
targets = [
    7 * np.asarray((np.sin(np.pi * 2 * t / NUM_TASK), np.cos(np.pi * 2 * t / NUM_TASK))) + (5, 5)
    for t in range(NUM_TASK)
]
losses = [Loss(t, a, coeff) for t, a in zip(targets, angles)]

plt.figure(dpi=200, figsize=(16, 16))
X = np.linspace(-5.0, 15.0, 100)
Y = np.linspace(-5.0, 15.0, 100)
XX, YY = np.meshgrid(X, Y)

Zs = [loss.get_contour(XX, YY) for loss in losses]
for t, Z in zip(targets, Zs):
    plt.contour(X, Y, Z, levels=np.linspace(0, 100, 100), alpha=0.3)
    plt.plot(*t, "ro", markersize=2)


plt.xlim(-5, 15)
plt.ylim(-5, 15)
plt.tight_layout()
plt.savefig("losses.png")
np.savez("env.npz", angles=angles, targets=targets, coeff=coeff)
