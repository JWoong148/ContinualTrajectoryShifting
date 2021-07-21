import os

import torch
import torch.nn as nn
from absl import app, flags, logging
from torchvision.models.utils import load_state_dict_from_url

from dataloader import get_dataloader
from models.get_model import get_model
from utils import InfIterator, Logger, accuracy, check_args, get_optimizer, get_scheduler

FLAGS = flags.FLAGS
# Training
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("train_steps", 10000, "Total training steps")
flags.DEFINE_enum("lr_schedule", "step_lr", ["step_lr", "cosine_lr"], "lr schedule")
flags.DEFINE_enum("opt", "adam", ["adam", "sgd", "rmsprop"], "optimizer")
flags.DEFINE_float("lr", 1e-3, "Learning rate")
flags.DEFINE_float("momentum", 0.9, "Momentum")
flags.DEFINE_float("weight_decay", 5e-4, "Weight decay")
flags.DEFINE_bool("nesterov", False, "Nesterov")

# Model
flags.DEFINE_string("model", "resnet20", "Model")
flags.DEFINE_bool("pretrained", False, "Start with pretrained weight")
flags.DEFINE_bool("finetune", False, "Finetune")

# Data
flags.DEFINE_integer("img_size", 32, "Image size")
flags.DEFINE_string("data", "stl10", "Data")

# Misc
flags.DEFINE_string("tblog_dir", None, "Directory for tensorboard logs")
flags.DEFINE_string("code_dir", "", "Directory for backup code")
flags.DEFINE_string("save_dir", "", "Directory for checkpoints")
flags.DEFINE_string("src_name", "", "Source name to use")
flags.DEFINE_string("src_steps", "10000", "Source training steps")
flags.DEFINE_string("exp_name", "", "Experiment name")
flags.DEFINE_integer("print_every", 200, "Print period")
flags.DEFINE_string("gpus", "", "GPUs to use")
flags.DEFINE_integer("num_workers", 3, "The number of workers for dataloading")


def train_step(model, train_iter, opt, device, criterion, logger):
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


def main(argv):
    del argv
    check_args(FLAGS)

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    os.environ["WANDB_SILENT"] = "true"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    train_loader, test_loader, num_classes = get_dataloader(
        name=FLAGS.data, img_size=FLAGS.img_size, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers
    )
    train_iter = InfIterator(train_loader)
    logging.info(f"Train dataset: {len(train_loader)} batches")
    logging.info(f"Total {FLAGS.train_steps//len(train_loader)} epochs")
    logging.info(f"Test dataset: {len(test_loader)} batches")

    # Model
    model = get_model(FLAGS.model, num_classes=num_classes).to(device)

    if FLAGS.src_name != "":
        state_dict = torch.load(f"{FLAGS.save_dir}/{FLAGS.src_name}/src/split_1/base_param_{FLAGS.src_steps}.pth")
        model.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Model is loaded from {FLAGS.save_dir}/{FLAGS.src_name}/src/split_1/base_param_{FLAGS.src_steps}.pth"
        )
    if FLAGS.finetune:
        if FLAGS.model == "resnet18":
            imagenet_pretrained = load_state_dict_from_url("https://download.pytorch.org/models/resnet18-5c106cde.pth")
            state_dict = {name: w for name, w in imagenet_pretrained.items() if w.requires_grad and "fc" not in name}
            model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = torch.load("checkpoints/scratch_tiny_imagenet_50k/tgt/model_50000.pth")
            _state_dict = {name: w for name, w in state_dict.items() if not ("running" in name or "fc" in name)}
            model.load_state_dict(_state_dict, strict=False)

    # Optimizer, scheduler, and criterion
    opt = get_optimizer(
        FLAGS.opt,
        model.parameters(),
        FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        nesterov=FLAGS.nesterov,
    )
    scheduler = get_scheduler(FLAGS.lr_schedule, opt, FLAGS.train_steps)
    criterion = nn.NLLLoss().to(device)

    # Logger
    logger = Logger(
        exp_name=FLAGS.exp_name,
        log_dir=FLAGS.log_dir,
        save_dir=FLAGS.save_dir,
        exp_suffix="tgt",
        print_every=FLAGS.print_every,
        save_every=FLAGS.train_steps,
        total_step=FLAGS.train_steps,
        use_wandb=True,
        wnadb_project_name="largemeta",
        wandb_config=FLAGS,
    )
    logger.register_model_to_save(model, "model")

    # Training Loop
    logger.start()
    for step in range(1, FLAGS.train_steps + 1):
        train_step(model, train_iter, opt, device, criterion, logger)
        scheduler.step()
        if step % FLAGS.print_every == 0:
            test(model, test_loader, device, criterion, logger)
        logger.step()
    logger.finish()


if __name__ == "__main__":
    app.run(main)
