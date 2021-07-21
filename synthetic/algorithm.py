from datetime import datetime

import numpy as np
import torch
from utils import (
    draw_loss_surface,
    draw_rs,
    draw_meta_path,
    draw_task_path,
    get_canvas,
    get_opt,
    get_env,
    draw_meta_loss_contour,
)

NUM_TASK, losses = get_env("env.npz")


def test(phi, lr, num_steps):
    thetas = [phi.clone().detach().requires_grad_(True) for _ in range(NUM_TASK)]
    opts = [get_opt("adam", t, lr) for t in thetas]
    hists = [[[phi[0].item()], [phi[1].item()]] for i in range(NUM_TASK)]

    tot_loss = 0
    for i in range(NUM_TASK):
        for step in range(num_steps):
            loss = losses[i](thetas[i])
            opts[i].zero_grad()
            loss.backward()
            opts[i].step()

            hists[i][0].append(thetas[i][0].item())
            hists[i][1].append(thetas[i][1].item())

        tot_loss += loss.item() / NUM_TASK
    return hists, tot_loss


def reptile(phi, inner_opt, lr, inner_step, meta_opt, meta_lr, meta_step, fig_dir):
    phi.grad = torch.zeros_like(phi)
    opt = get_opt(meta_opt, phi, meta_lr)

    hist = [[phi[0].item()], [phi[1].item()]]
    for meta_step in range(1, meta_step + 1):
        thetas = [phi.clone().detach().requires_grad_(True) for _ in range(NUM_TASK)]
        opts = [get_opt(inner_opt, t, lr) for t in thetas]
        for step in range(1, inner_step + 1):
            for i in range(NUM_TASK):
                loss = losses[i](thetas[i])
                opts[i].zero_grad()
                loss.backward()
                opts[i].step()

        meta_grad = torch.zeros(2)
        for i in range(NUM_TASK):
            meta_grad += thetas[i] / NUM_TASK

        phi.grad.copy_(phi - meta_grad)
        opt.step()

        hist[0].append(phi[0].item())
        hist[1].append(phi[1].item())

    hists, tot_loss = test(phi, lr, inner_step)
    title = f"reptile_M_{meta_step}_K_{inner_step}_{tot_loss:.2f}"

    meta_losses = np.load("meta_losses_100.npy")

    fig, ax = get_canvas()
    draw_meta_loss_contour(fig, ax, meta_losses, inner_step, 100)
    draw_meta_path(ax, hist)
    ax.set_title(title)
    fig.savefig(f"{fig_dir}/{title}.png")

    fig, ax = get_canvas()
    draw_rs(fig, ax, meta_losses, 100)
    draw_meta_path(ax, hist)
    ax.set_title(title)
    fig.savefig(f"{fig_dir}/{title}_rs.png")

    for tidx in range(8):
        fig, ax = get_canvas()
        draw_rs(fig, ax, meta_losses, 100)
        draw_loss_surface(fig, ax, [losses[tidx]])
        draw_meta_path(ax, hist)
        draw_task_path(ax, hists, tidx)
        ax.set_title(title)
        fig.savefig(f"{fig_dir}/{title}_tp_{tidx}.png")


def ours_accurate(phi, inner_opt, lr, inner_step, meta_opt, meta_lr, meta_step, fig_dir):
    phi.grad = torch.zeros_like(phi)
    opt = get_opt(meta_opt, phi, meta_lr)

    hist = [[phi[0].item()], [phi[1].item()]]
    for meta_step in range(1, meta_step + 1):
        thetas = [phi.clone().detach().requires_grad_(True) for _ in range(NUM_TASK)]
        opts = [get_opt(inner_opt, t, lr) for t in thetas]
        for _step in range(1, inner_step + 1):
            for step in range(1, _step + 1):
                for i in range(NUM_TASK):
                    loss = losses[i](thetas[i])
                    opts[i].zero_grad()
                    loss.backward()
                    opts[i].step()

            meta_grad = torch.zeros(2)
            for i in range(NUM_TASK):
                meta_grad += thetas[i] / NUM_TASK

            phi.grad.copy_(phi - meta_grad)
            opt.step()

            hist[0].append(phi[0].item())
            hist[1].append(phi[1].item())

    hists, tot_loss = test(phi, lr, inner_step)
    title = f"ours_accurate_M_{meta_step}_K_{inner_step}_{tot_loss:.2f}"

    print(title, et)
    meta_losses = np.load("meta_losses_100.npy")

    fig, ax = get_canvas()
    draw_meta_loss_contour(fig, ax, meta_losses, inner_step, 100)
    draw_meta_path(ax, hist)
    ax.set_title(title)
    fig.savefig(f"{fig_dir}/{title}.png")

    fig, ax = get_canvas()
    draw_rs(fig, ax, meta_losses, 100)
    draw_meta_path(ax, hist)
    ax.set_title(title)
    fig.savefig(f"{fig_dir}/{title}_rs.png")

    for tidx in range(8):
        fig, ax = get_canvas()
        draw_rs(fig, ax, meta_losses, 100)
        draw_loss_surface(fig, ax, [losses[tidx]])
        draw_meta_path(ax, hist)
        draw_task_path(ax, hists, tidx)
        ax.set_title(title)
        fig.savefig(f"{fig_dir}/{title}_tp_{tidx}.png")


def ours(phi, inner_opt, lr, inner_step, meta_opt, meta_lr, meta_step, fig_dir):
    st = datetime.now()
    phi.grad = torch.zeros_like(phi)
    opt = get_opt(meta_opt, phi, meta_lr, 0.1)

    hist = [[phi[0].item()], [phi[1].item()]]
    for meta_step in range(1, meta_step + 1):
        thetas = [phi.clone().detach().requires_grad_(True) for _ in range(NUM_TASK)]
        opts = [get_opt(inner_opt, t, lr) for t in thetas]
        for step in range(1, inner_step + 1):
            for i in range(NUM_TASK):
                loss = losses[i](thetas[i])
                opts[i].zero_grad()
                loss.backward()
                opts[i].step()

            _phi = phi.clone().detach()
            meta_grad = torch.zeros(2)
            for i in range(NUM_TASK):
                meta_grad += thetas[i] / NUM_TASK

            phi.grad.copy_(phi - meta_grad)
            opt.step()
            with torch.no_grad():
                for theta in thetas:
                    theta.add_(phi - _phi)

            hist[0].append(phi[0].item())
            hist[1].append(phi[1].item())
    return hist
