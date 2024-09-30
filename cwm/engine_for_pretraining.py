from typing import Iterable
import torch
import torch.nn as nn
from timm.data.constants import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
import utils

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    args=None,
                    loss_func = nn.MSELoss(),
    ):

    metric_logger = utils.MetricLogger(delimiter="  ")
    model.train()
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f'Epoch [{epoch}]'
    patch_size = model.module.encoder.patch_size[-2:]
    tubelet_size = model.module.encoder.patch_size[0]
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]

    for step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        # assign learning rate & weight decay for each iteration
        it = start_steps + step  # global training iteration
        if (lr_schedule_values is not None or wd_schedule_values is not None) and (step % args.accum_iter == 0):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # prepare input
        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1)

        # prepare target
        with torch.no_grad():
            unnorm_videos = videos * std + mean  # in [0, 1]
            videos_patch = utils.patchify(unnorm_videos, tubelet_size, patch_size)
            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        # feedforward
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        # backward
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=model.parameters(), create_graph=is_second_order,
                    update_grad=(step + 1) % args.accum_iter == 0)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        if (step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
