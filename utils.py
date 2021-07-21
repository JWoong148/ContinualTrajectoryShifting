import os
import shutil
from datetime import datetime

import torch
import torch.optim as optim
import wandb
from absl import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.distributed as dist


def get_optimizer(opt_name, params, lr, momentum=0.9, weight_decay=5e-4, nesterov=False):
    if opt_name == "adam":
        opt = optim.Adam(params, lr=lr, weight_decay=5e-4)
    elif opt_name == "rmsprop":
        opt = optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "sgd":
        opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise NotImplementedError
    return opt


def get_scheduler(scheduler_name, opt, train_steps, milestones=[0.4, 0.7, 0.9], gamma=0.3):
    if scheduler_name == "step_lr":
        milestones = [int(train_steps * v) for v in milestones]
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
    elif scheduler_name == "cosine_lr":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, train_steps)
    else:
        raise NotImplementedError
    return scheduler


def share_grads(params):
    tensors = torch.cat([p.grad.view(-1) for p in params])
    dist.all_reduce(tensors)
    tensors /= dist.get_world_size()

    idx = 0
    for p in params:
        p.grad.data.copy_(tensors[idx : idx + np.prod(p.shape)].view(p.size()))
        idx += np.prod(p.shape)


def share_params(params):
    tensors = torch.cat([p.view(-1) for p in params])
    dist.all_reduce(tensors)
    tensors /= dist.get_world_size()

    idx = 0
    for p in params:
        p.data.copy_(tensors[idx : idx + np.prod(p.shape)].view(p.size()))
        idx += np.prod(p.shape)


def accuracy(y, y_pred):
    with torch.no_grad():
        pred = torch.max(y_pred, dim=1)
        return 1.0 * pred[1].eq(y).sum() / y.size(0)


class InfIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)


def check_args(FLAGS):
    ignore = [
        "logtostderr",
        "alsologtostderr",
        "log_dir",
        "v",
        "verbosity",
        "stderrthreshold",
        "showprefixforinfo",
        "run_with_pdb",
        "pdb_post_mortem",
        "run_with_profiling",
        "profile_file",
        "use_cprofile_for_profiling",
        "only_check_args",
        "?",
        "help",
        "helpshort",
        "helpfull",
        "helpxml",
    ]
    for name, value in FLAGS.flag_values_dict().items():
        if name not in ignore:
            print(f"{name:>20} : {value}")
    # print("Is this correct? (y/n)")
    # ret = input()
    # if ret.lower() != "y":
    #     exit(0)


def backup_code(
    backup_dir,
    ignore_list={
        ".gitignore",
        ".ipynb_checkpoints",
        "codes",
        ".vscode",
        "__pycache__",
        "checkpoints",
        "data",
        "runs",
        "wandb",
    },
):
    shutil.copytree(
        os.path.abspath(os.path.curdir), backup_dir, ignore=lambda src, names: ignore_list,
    )


class Logger:
    def __init__(
        self,
        exp_name,
        exp_suffix="",
        log_dir=None,
        save_dir=None,
        print_every=100,
        save_every=100,
        initial_step=0,
        total_step=0,
        print_to_stdout=True,
        use_wandb=False,
        wnadb_project_name=None,
        wandb_tags=[],
        wandb_config=None,
    ):
        if log_dir is not None and log_dir != "":
            self.log_dir = os.path.join(log_dir, exp_name, exp_suffix)
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = None
            assert use_wandb, "'log_dir' argument must be given or 'use_wandb' argument must be True."

        if save_dir is not None:
            self.save_dir = os.path.join(save_dir, exp_name, exp_suffix)
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.print_every = print_every
        self.save_every = save_every
        self.step_count = initial_step
        self.total_step = total_step
        self.print_to_stdout = print_to_stdout
        self.use_wandb = use_wandb

        self.writer = None
        self.start_time = None
        self.groups = dict()
        self.models_to_save = dict()
        self.objects_to_save = dict()
        if self.use_wandb:
            if "/" in exp_suffix:
                exp_suffix = "_".join(exp_suffix.split("/")[:-1])
            wandb.init(project=wnadb_project_name, name=exp_name + "_" + exp_suffix, tags=wandb_tags, reinit=True)
            wandb.config.update(wandb_config)

    def register_model_to_save(self, model, name):
        assert name not in self.models_to_save.keys(), "Name is already registered."

        self.models_to_save[name] = model

    def register_object_to_save(self, object, name):
        assert name not in self.objects_to_save.keys(), "Name is already registered."

        self.objects_to_save[name] = object

    def step(self):
        self.step_count += 1
        if self.step_count % self.print_every == 0:
            if self.print_to_stdout:
                self.print_log(self.step_count, self.total_step, elapsed_time=datetime.now() - self.start_time)
            self.write_log(self.step_count)

        if self.step_count % self.save_every == 0:
            self.save_models(self.step_count)
            self.save_objects(self.step_count)

    def meter(self, group_name, log_name, value):
        if group_name not in self.groups.keys():
            self.groups[group_name] = dict()

        if log_name not in self.groups[group_name].keys():
            self.groups[group_name][log_name] = Accumulator()

        self.groups[group_name][log_name].update_state(value)

    def reset_state(self):
        for _, group in self.groups.items():
            for _, log in group.items():
                log.reset_state()

    def print_log(self, step, total_step, elapsed_time=None):
        print(f"[Step {step:5d}/{total_step}]", end="  ")

        for name, group in self.groups.items():
            print(f"({name})", end="  ")
            for log_name, log in group.items():
                if "acc" in log_name.lower():
                    print(f"{log_name} {log.result() * 100:.2f}", end=" | ")
                else:
                    print(f"{log_name} {log.result():.4f}", end=" | ")

        if elapsed_time is not None:
            print(f"(Elapsed time) {elapsed_time}")
        else:
            print()

    def write_log(self, step):
        if self.use_wandb:
            log_dict = {}
            for group_name, group in self.groups.items():
                for log_name, log in group.items():
                    log_dict["{}/{}".format(log_name, group_name)] = log.result()
            wandb.log(log_dict, step=step)
        else:
            if self.writer is None:
                self.writer = SummaryWriter(self.log_dir)

            for group_name, group in self.groups.items():
                for log_name, log in group.items():
                    self.writer.add_scalar("{}/{}".format(log_name, group_name), log.result(), step)
            self.writer.flush()

        self.reset_state()

    def write_log_individually(self, name, value, step):
        if self.use_wandb:
            wandb.log({name: value}, step=step)
        else:
            self.writer.add_scalar(name, value, step=step)

    def save_models(self, suffix=None):
        if self.save_dir is None:
            return

        for name, model in self.models_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"{_name}.pth"))

            if self.print_to_stdout:
                logging.info(f"{name} is saved to {self.save_dir}")

    def save_objects(self, suffix=None):
        if self.save_dir is None:
            return

        for name, obj in self.objects_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(obj, os.path.join(self.save_dir, f"{_name}.pth"))

            if self.print_to_stdout:
                logging.info(f"{name} is saved to {self.save_dir}")

    def start(self):
        if self.print_to_stdout:
            logging.info("Training starts!")
        self.save_models("init")
        self.save_objects("init")
        self.start_time = datetime.now()

    def finish(self):
        if self.step_count % self.save_every != 0:
            self.save_models(self.step_count)
            self.save_objects(self.step_count)

        if self.print_to_stdout:
            logging.info("Training is finished!")

        if self.use_wandb:
            wandb.join()


class Accumulator:
    def __init__(self):
        self.data = 0
        self.num_data = 0

    def reset_state(self):
        self.data = 0
        self.num_data = 0

    def update_state(self, tensor):
        with torch.no_grad():
            self.data += tensor
            self.num_data += 1

    def result(self):
        if self.num_data == 0:
            return 0
        return (1.0 * self.data / self.num_data).item()
