import os
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
from absl import app, flags, logging
from torch.multiprocessing import Process
from torchvision.models.utils import load_state_dict_from_url

from dataloader import get_dataloader
from models.get_model import get_model
from utils import InfIterator, Logger, accuracy, backup_code, check_args, get_optimizer, share_params, share_grads
from leap import Updater, clone_state_dict

FLAGS = flags.FLAGS
# Training
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("train_steps", 50000, "Total training steps for a single run")
flags.DEFINE_enum("opt", "sgd", ["adam", "sgd", "rmsprop"], "optimizer")
# flags.DEFINE_enum("meta_opt", "sgd_pure", ["adam", "sgd", "rmsprop"], "meta optimizer")
flags.DEFINE_float("lr", 1e-2, "Learning rate")
flags.DEFINE_float("meta_lr", 1e-2, "Meta learning rate")
flags.DEFINE_integer("aggregate_period", 1, "Aggregate period")
flags.DEFINE_integer("reset_period", 1000, "Reset period (K)")
flags.DEFINE_float("momentum", 0.9, "Momentum")
flags.DEFINE_float("weight_decay", 5e-4, "Weight decay")
flags.DEFINE_bool("nesterov", False, "Nesterov")
flags.DEFINE_bool("hard_reset", False, "Reset model & opt or not")

# Model
flags.DEFINE_string("model", "resnet20", "Model")
flags.DEFINE_bool("pretrained", False, "Start with pretrained weight")

# Data

# Misc
flags.DEFINE_string("tblog_dir", None, "Directory for tensorboard logs")
flags.DEFINE_string("code_dir", "", "Directory for backup code")
flags.DEFINE_string("save_dir", "", "Directory for checkpoints")
flags.DEFINE_string("exp_name", "", "Experiment name")
flags.DEFINE_integer("print_every", 500, "Print period")
flags.DEFINE_integer("save_every", 5000, "Save period")
flags.DEFINE_list("gpus", "", "GPUs to use")
flags.DEFINE_string("port", "123456", "Port number for multiprocessing")
flags.DEFINE_integer("num_workers", 1, "The number of workers for dataloading")

SRC_DATA = {
    "name": ["birdsnap", "sun397", "dtd", "vgg_pets", "stanford_40_actions", "fungi", "fruit_360", "deepweeds"],
    "img_size": [224, 224, 224, 224, 224, 224, 100, 224],
}


def train_step(model, updater, train_iter, opt, device, criterion, logger):
    model.train()

    # Sample data from training set
    x, y = next(train_iter)
    x, y = x.to(device), y.to(device)
    x = x.expand(-1, 3, -1, -1)

    # Update theta
    y_pred = nn.LogSoftmax(dim=1)(model(x))
    loss = criterion(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    curr_state = clone_state_dict(model.state_dict(keep_vars=True))
    curr_loss = loss.clone()
    updater(curr_loss, curr_state)

    # Meter logs
    logger.meter("train", "ce_loss", loss)
    logger.meter("train", "accuracy", accuracy(y, y_pred))


def test(model, test_loader, device, criterion, logger):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.expand(-1, 3, -1, -1)

            y_pred = nn.LogSoftmax(dim=1)(model(x))
            loss = criterion(y_pred, y)
            logger.meter("test", "ce_loss", loss)

            pred = torch.max(y_pred, dim=1)
            correct += pred[1].eq(y).sum()
            total += y.size(0)
    logger.meter("test", "accuracy", 1.0 * correct / total)


def run_single_process(rank, backend="nccl"):
    dist.init_process_group(backend, rank=rank, world_size=8)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus[rank]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    train_loader, test_loader, num_classes = get_dataloader(
        name=SRC_DATA["name"][rank],
        img_size=SRC_DATA["img_size"][rank],
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )
    train_iter = InfIterator(train_loader)
    msg = f"\n[rank {rank}]\n"
    msg += f"Dataset: {SRC_DATA['name'][rank]}\n"
    msg += f"Image size: {SRC_DATA['img_size'][rank]}\n"
    msg += f"Train dataset: {len(train_loader)} batches\n"
    msg += f"Total {FLAGS.train_steps//len(train_loader)} epochs\n"
    msg += f"Test dataset: {len(test_loader)} batches\n"
    logging.info(msg)

    # Model
    model = get_model(FLAGS.model, num_classes=num_classes).to(device)
    if FLAGS.pretrained:
        if FLAGS.model == "resnet18":
            imagenet_pretrained = load_state_dict_from_url("https://download.pytorch.org/models/resnet18-5c106cde.pth")
            state_dict = {name: w for name, w in imagenet_pretrained.items() if w.requires_grad and "fc" not in name}
            model.load_state_dict(state_dict, strict=False)

    # Synchronize phi at the beginning
    share_params(model.parameters())
    base_param = {
        name: w.clone().detach().requires_grad_(True)
        for name, w in model.named_parameters()
        if w.requires_grad and "fc" not in name
    }
    for p in base_param.values():
        if p.grad is None:
            p.grad = p.new(*p.shape)
        p.grad.zero_()
    updater = Updater(base_param)

    # Criterion & Optimizers
    criterion = nn.NLLLoss().to(device)
    opt = get_optimizer(
        FLAGS.opt,
        model.parameters(),
        FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        nesterov=FLAGS.nesterov,
    )

    # Logger
    logger = Logger(
        exp_name=FLAGS.exp_name,
        log_dir=FLAGS.log_dir,
        save_dir=FLAGS.save_dir,
        exp_suffix=f"src/split_{rank+1}",
        print_every=FLAGS.print_every,
        save_every=FLAGS.save_every,
        total_step=FLAGS.train_steps,
        print_to_stdout=(rank == 0),
        use_wandb=True,
        wnadb_project_name="largemeta",
        wandb_tags=[f"split_{rank+1}"],
        wandb_config=FLAGS,
    )
    # logger.register_model_to_save(model, "model")
    if rank == 0:
        logger.register_object_to_save(base_param, "base_param")

    # Training Loop
    logger.start()
    for i in range(1, FLAGS.train_steps + 1):

        train_step(model, updater, train_iter, opt, device, criterion, logger)
        if i % FLAGS.print_every == 0:
            test(model, test_loader, device, criterion, logger)
        logger.step()

        if i % FLAGS.reset_period == 0:
            share_grads(base_param.values())
            for p in base_param.values():
                p.data.add_(p.grad, alpha=-FLAGS.meta_lr)
                p.grad.zero_()
            if FLAGS.hard_reset:
                model = get_model(FLAGS.model, num_classes=num_classes).to(device)
                opt = get_optimizer(
                    FLAGS.opt,
                    model.parameters(),
                    FLAGS.lr,
                    momentum=FLAGS.momentum,
                    weight_decay=FLAGS.weight_decay,
                    nesterov=FLAGS.nesterov,
                )
            model.load_state_dict(base_param, strict=False)
            updater.initialize()

    logger.finish()


def run_multi_process(argv):
    del argv
    check_args(FLAGS)
    backup_code(os.path.join(FLAGS.code_dir, FLAGS.exp_name, datetime.now().strftime("%m-%d-%H-%M-%S")))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = FLAGS.port
    os.environ["WANDB_SILENT"] = "true"
    processes = []

    for rank in range(len(SRC_DATA["name"])):
        p = Process(target=run_single_process, args=(rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    app.run(run_multi_process)
