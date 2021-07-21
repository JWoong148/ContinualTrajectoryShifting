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

FLAGS = flags.FLAGS
# Training
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("train_steps", 500000, "Total training steps for a single run")
flags.DEFINE_enum("opt", "sgd", ["adam", "sgd", "rmsprop"], "optimizer")
flags.DEFINE_float("lr", 1e-2, "Learning rate")
flags.DEFINE_integer("aggregate_period", 1, "Aggregate period")
flags.DEFINE_integer("reset_period", 1000, "Reset period (K)")
flags.DEFINE_float("momentum", 0.9, "Momentum")
flags.DEFINE_float("weight_decay", 5e-4, "Weight decay")
flags.DEFINE_bool("nesterov", False, "Nesterov")
flags.DEFINE_bool("hard_reset", False, "Reset model & opt or not")

flags.DEFINE_enum("meta_opt", "sgd", ["adam", "sgd", "rmsprop"], "optimizer")
flags.DEFINE_float("meta_lr", 0.1, "Meta learning rate")
flags.DEFINE_float("meta_momentum", 0, "Momentum")
flags.DEFINE_float("meta_weight_decay", 0, "Weight decay")
flags.DEFINE_bool("meta_nesterov", False, "Nesterov")

flags.DEFINE_float("lamb", 0.5, "lamb")
flags.DEFINE_float("cg_damp", 0, "lamb")
flags.DEFINE_integer("val_steps", 10, "Val steps")
flags.DEFINE_integer("cg_iterations", 10, "CG iterations")

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
    "name": [
        "tiny_imagenet_split_1",
        "tiny_imagenet_split_2",
        "cifar_100",
        "stanford_dogs",
        "aircraft",
        "cub",
        "fashion_mnist",
        "svhn",
    ],
    "img_size": [64, 64, 32, 84, 84, 84, 28, 32],
}
# SRC_DATA = {
#     "name": ["birdsnap", "sun397", "dtd", "vgg_pets", "stanford_40_actions", "fungi", "fruit_360", "deepweeds"],
#     "img_size": [224, 224, 224, 224, 224, 224, 100, 224],
# }


def hessian_vector_product(grad, v, params):
    hv = torch.autograd.grad(grad, params, retain_graph=True, grad_outputs=v.clone().detach())
    hv = torch.cat([g.contiguous().view(-1) for g in hv])
    return hv


def get_Av(grad, params, lamb):
    def Av(v):
        hvp = hessian_vector_product(grad, v, params)
        return v + hvp / lamb

    return Av


def conjugate_gradient(Av, b, x, num_iterations):
    Ax = Av(x) + FLAGS.cg_damp * x

    r = b - Ax
    p = r.clone().detach()

    for _ in range(num_iterations):
        Ap = Av(p) + FLAGS.cg_damp * p
        rTr = r.dot(r)
        alpha = rTr / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rTr_new = r.dot(r)
        beta = rTr_new / rTr
        p = r + beta * p

    return x


def train_step(model, train_iter, val_iter, opt, base_param, compute_meta_grad, device, criterion, logger):
    model.train()

    # Sample data from training set
    x, y = next(train_iter)
    x, y = x.to(device), y.to(device)
    x = x.expand(-1, 3, -1, -1)

    # Update theta
    y_pred = nn.LogSoftmax(dim=1)(model(x))
    ce_loss = criterion(y_pred, y)
    reg_loss = 0
    # for name, w in model.named_parameters():
    #     if not w.requires_grad or "fc" in name:
    #         continue
    #     reg_loss += 0.5 * FLAGS.lamb * torch.sum((base_param[name] - w) ** 2)
    loss = ce_loss + reg_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    # Meter logs
    logger.meter("train", "ce_loss", loss)
    logger.meter("train", "accuracy", accuracy(y, y_pred))

    if compute_meta_grad:
        params = [w for name, w in model.named_parameters() if "fc" not in name and w.requires_grad]

        tr_loss = criterion(nn.LogSoftmax(dim=1)(model(x)), y)
        tr_grad = torch.autograd.grad(tr_loss, params, create_graph=True)
        tr_grad = torch.cat([g.contiguous().view(-1) for g in tr_grad])
        # tr_grad = torch.nn.utils.convert_parameters.parameters_to_vector(tr_grad)

        val_grad = 0
        for _ in range(FLAGS.val_steps):
            x_val, y_val = next(val_iter)
            x_val, y_val = x_val.to(device), y_val.to(device)
            x_val = x_val.expand(-1, 3, -1, -1)

            val_loss = criterion(nn.LogSoftmax(dim=1)(model(x_val)), y_val) / FLAGS.val_steps
            _val_grad = torch.autograd.grad(val_loss, params)
            _val_grad = torch.cat([g.contiguous().view(-1) for g in _val_grad])
            val_grad += _val_grad

        # val_grad = torch.nn.utils.convert_parameters.parameters_to_vector(val_grad)
        # implicit_grad = val_grad.clone()
        Av = get_Av(tr_grad, params, FLAGS.lamb)
        implicit_grad = conjugate_gradient(
            Av, val_grad.clone().detach(), torch.zeros_like(val_grad), FLAGS.cg_iterations
        )

        idx = 0
        for p in base_param.values():
            p.data -= FLAGS.meta_lr * implicit_grad[idx : idx + p.numel()].view_as(p)
            idx += p.numel()
        assert idx == implicit_grad.numel()
        # for name, w in model.named_parameters():
        #     if not w.requires_grad or "fc" in name:
        #         continue
        #     base_param[name].data -= FLAGS.meta_lr * w.grad
        share_params(base_param.values())
        # if base_param[name].grad is None:
        #     base_param[name].grad = w.grad.data
        # else:
        #     base_param[name].grad.copy_(w.grad.data)

        # share_grads(base_param.values())


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
    train_loader, val_loader, test_loader, num_classes = get_dataloader(
        name=SRC_DATA["name"][rank],
        img_size=SRC_DATA["img_size"][rank],
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )
    train_iter = InfIterator(train_loader)
    val_iter = train_iter
    # val_iter = InfIterator(val_loader)
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
        name: w.clone().detach() for name, w in model.named_parameters() if w.requires_grad and "fc" not in name
    }

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
    # meta_opt = get_optimizer(
    #     FLAGS.meta_opt,
    #     base_param.values(),
    #     FLAGS.meta_lr,
    #     momentum=FLAGS.meta_momentum,
    #     weight_decay=FLAGS.meta_weight_decay,
    #     nesterov=FLAGS.meta_nesterov,
    # )

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
        train_step(model, train_iter, val_iter, opt, base_param, i % FLAGS.reset_period == 0, device, criterion, logger)
        if i % FLAGS.print_every == 0:
            test(model, test_loader, device, criterion, logger)
        logger.step()

        if i % FLAGS.reset_period == 0:
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
            train_loader, val_loader, test_loader, num_classes = get_dataloader(
                name=SRC_DATA["name"][rank],
                img_size=SRC_DATA["img_size"][rank],
                batch_size=FLAGS.batch_size,
                num_workers=FLAGS.num_workers,
            )
            train_iter = InfIterator(train_loader)
            val_iter = InfIterator(val_loader)

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
        p = Process(target=run_single_process, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    app.run(run_multi_process)
