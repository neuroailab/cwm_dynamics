
import argparse
import datetime
import numpy as np
import random
import time
import torch
import json
import os
from pathlib import Path
from optim_factory import create_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from cwm.data.dataset_utils import build_pretraining_dataset
from cwm.model import model_pretrain
from engine_for_pretraining import train_one_epoch
import wandb
import torch.backends.cudnn as cudnn
np.random.seed(0)
random.seed(0)

def get_args():
    parser = argparse.ArgumentParser('CWM pre-training script', add_help=False)

    # training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='per-GPU batch-size')
    parser.add_argument('--epochs', default=800, type=int, help='number of training epochs')
    parser.add_argument('--save_ckpt_freq', default=50, type=int, help='save checkpoint frequency')
    parser.add_argument('--print_freq', default=1, type=int, help='frequency of printing training stats')
    parser.add_argument('--accum_iter', default=1, type=int, help='number of steps to accumulate gradients')
    parser.add_argument('--inference', action='store_true', help='evaluation mode')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--val_after', default=50, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')

    # Model parameters
    parser.add_argument('--model', default='vitb_8x8patch_3frames', type=str, help='Name of model to train')
    parser.add_argument('--context_frames', type=int, default=2, help='number of frames model will see densely')
    parser.add_argument('--target_frames', type=int, default=1, help='number of frames model will see sparsely')
    parser.add_argument('--temporal_units', type=str, default='ms', help='the units in which time is defined')
    parser.add_argument('--sampling_rate', type=int, default=150, help='temporal gap between context/target frames')
    parser.add_argument('--context_target_gap', type=int, nargs='+', default=[150, 150], help='gap between context/target')

    # Masking and target parameters
    parser.add_argument('--mask_type', default='rotated_table', type=str, help='masked strategy')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='masking ratio')
    parser.add_argument('--mask_kwargs', default='', type=json.loads, help='extra arguments for masking generator')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default:adamw)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer epsilon')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=0.05, help='Final value of the weight decay.')
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR', help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR', help='lower lr bound for cyclic schedulers)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='steps to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str, help='dataset path')
    parser.add_argument('--data_path_list', type=str, nargs='+', default=None, help='[path1, path2, path3, ...]')
    parser.add_argument('--num_workers', default=10, type=int)

    # Augmentation parameters
    parser.add_argument('--augmentation_type', type=str, default='multiscale', choices=['multiscale', 'center', 'none'])
    parser.add_argument('--augmentation_scales', type=float, nargs='+', default=[1.0, 0.875, 0.75, 0.66])


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


# Assuming 'model' is your PyTorch model
def export_model_parameters(model):
    with open('model_parameters.txt', 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"{name} {param.size()}\n")


def main(args):
    ## Setup distributed training
    utils.init_distributed_mode(args)
    cudnn.benchmark = True
    device = torch.device(args.device)
    num_tasks = utils.get_world_size()
    sampler_rank = global_rank = utils.get_rank()
    world_size = utils.get_world_size()

    ## Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## Initialize model
    model = getattr(model_pretrain, args.model)()
    args.input_size = int(model.encoder.patch_embed.img_size[0])
    args.tubelet_size = model.patch_size[0]

    args.mask_input_size = (
        (args.context_frames + args.target_frames) // args.tubelet_size,
        args.input_size // model.patch_size[-2],
        args.input_size // model.patch_size[-1],
    )

    ## Prepare datasets
    dataset_train = build_pretraining_dataset(args)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=sampler_rank,
        shuffle=True,
        drop_last=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        worker_init_fn=utils.seed_worker,
    )

    num_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

    n_params, n_params_str = utils.get_model_num_parameters(model)

    total_batch_size = args.batch_size * world_size * args.accum_iter

    ## LR and warmup
    export_model_parameters(model)

    model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=False)

    ## Optimizer, loss scaler
    optimizer = create_optimizer(args, model.module)
    loss_scaler = NativeScaler()

    ## LR scheduler, WD scheduler
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_steps_per_epoch
    )

    ## Resume from checkpoint, if any
    utils.auto_load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    ## Print training arguments
    print("world size: %d" % args.world_size)
    print("model: %s" % args.model)
    print("image size: %s" % str(args.input_size))
    print("patch size: %s" % str(model.module.encoder.patch_embed.patch_size[-2:]))
    print("context frames: %s" % str(args.context_frames))
    print("target frames: %s" % str(args.target_frames))
    print("per-device batch size: %d" % total_batch_size)
    print("total batch size: %d" % total_batch_size)
    print("grad accumulation: %d" % args.accum_iter)
    print("dataset length: %d" % len(dataset_train))
    print("steps per epoch: %d" % num_steps_per_epoch)
    print("num parameters: %s" % n_params_str)
    print("lr: %.8f" % args.lr)

    ## Setup logging
    if args.use_wandb and utils.is_main_process():
        wandb.init(project="cwm", name=args.output_dir.split('/')[-1], config=args)


    print(f'start training at epoch {args.start_epoch} for {args.epochs} epochs')
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # Run one epoch
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            start_steps=epoch * num_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            args=args
        )

        # Save checkpoint
        if args.output_dir and ((epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            utils.save_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        # Logging
        start_time = time.time()
        do_write = utils.is_main_process()
        if args.output_dir and do_write:
            log_stats = {
                **{f'train/{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'params': n_params,
                'epoch_time': time.time() - start_time
            }

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.use_wandb:
                wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    main(opts)
