import datetime
import io
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path
import contextlib
import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import get_state_dict
from torch import inf
from skimage import measure
from PIL import Image
import sys

try:
    sys.path.append(os.path.join(os.environ['HOME'], '.cache/torch/CutLER'))
    sys.path.append(os.path.join(os.environ['HOME'], '.cache/torch/CutLER/maskcut'))
    sys.path.append(os.path.join(os.environ['HOME'], '.cache/torch/CutLER/third_party'))
    import dino
    from maskcut import get_affinity_matrix, second_smallest_eigenvector, get_salient_areas, check_num_fg_corners, get_masked_affinity_matrix
    global dino_backbone
    dino_backbone = None
except:
    pass

def patchify(x, tubelet_size, patch_size):
    '''
    :param x: [B, C, T, H, W]
    :param tubelet_size: 2
    :param patch_size: (8, 8)
    :return:
    '''
    videos_squeeze = rearrange(x,
                               'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                               p0=tubelet_size,
                               p1=patch_size[0],
                               p2=patch_size[1])

    videos_patch = rearrange(videos_squeeze, 'b n p c -> b n (p c)')

    return videos_patch

def imagenet_unnormalize(x, temporal_dim=2):
    device = x.device

    if len(x.shape) == 3:
        if x.shape[0] == 3:  # "channel_first"
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[:, None, None].to(x)
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[:, None, None].to(x)
        else:  # channel_last
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, None, :].to(x)
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, None, :].to(x)
    elif len(x.shape) == 4:
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None].to(x)
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None].to(x)
    elif len(x.shape) == 5:
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, None, :, None, None].to(x)
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, None, :, None, None].to(x)

        if temporal_dim == 2:
            mean = mean.transpose(1, 2)
            std = std.transpose(1, 2)

    return x * std + mean

def imagenet_normalize(x, temporal_dim=2):
    device = x.device

    if len(x.shape) == 3:
        if x.shape[0] == 3:  # "channel_first"
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[:, None, None].to(x)
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[:, None, None].to(x)
        else:  # channel_last
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, None, :].to(x)
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, None, :].to(x)
    elif len(x.shape) == 4:
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None].to(x)
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None].to(x)
    elif len(x.shape) == 5:
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, None, :, None, None].to(x)
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, None, :, None, None].to(x)

        if temporal_dim == 2:
            mean = mean.transpose(1, 2)
            std = std.transpose(1, 2)

    return (x - mean) / std

def sinusoidal_embedding(x, n_freq=5, keep_ori=True):
    """
    create sin embedding for 3d vectors
    input:
        x: *x3
        n_freq: number of raised frequency
    """

    shape = list(x.shape)
    assert x.shape[-1] == 3, "expect the last dimension to have size 3"
    x = x.reshape(-1, 3)

    embedded = []
    if keep_ori:
        embedded.append(x)
    emb_fns = [torch.sin, torch.cos]
    freqs = 2. ** torch.linspace(0., n_freq - 1, steps=n_freq)
    for freq in freqs:
        for emb_fn in emb_fns:
            embedded.append(emb_fn(freq * x))
    embedded = torch.cat(embedded, dim=-1)
    C = embedded.shape[-1]
    embedded = embedded.reshape(shape[:-1] + [C])
    return embedded

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update2(self, kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)


    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.2f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))





def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.gpu) # this is important for avoiding extra memory usage in GPU 0
    args.device = torch.device(f'cuda:{args.gpu}')
    args.dist_backend = 'nccl'
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        # breakpoint()
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    iter_per_len = iters/len(iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iter_per_len))
    # schedule = np.array(
    #     [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_model_num_parameters(model):

    num_parameters = sum([v.numel() for v in model.parameters() if v.requires_grad])

    human_readable_fn = lambda num: \
        f'{num / 1e9:.3f} B' if num >= 1e9 else f'{num / 1e6:.3f} M' \
            if num >= 1e6 else f'{num / 1e3:.3f} K' if num >= 1e3 else str(num)
    num_parameters_str = human_readable_fn(num_parameters)

    return num_parameters, num_parameters_str

def save_model(args, epoch, model, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, optimizer, loss_scaler, model_ema=None, global_rank=None):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if len(args.resume) == 0:
            import glob
            if global_rank is None:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, f'checkpoint-*-rank-{global_rank}.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                if global_rank is None:
                    t = ckpt.split('-')[-1].split('.')[0]
                else:
                    t = ckpt.split('checkpoint-')[1].split('-')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                if global_rank is None:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
                else:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d-rank-%d.pth' % (latest_ckpt, global_rank))
            if args.resume:
                print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model.module.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])

    else:
        # deepspeed, only support '--auto_resume'.
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
            print("Auto resume checkpoint: %d" % latest_ckpt)
            _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
            args.start_epoch = client_states['epoch'] + 1
            if model_ema is not None:
                if args.model_ema:
                    _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2*3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def unpatchify_cwm(x, patch_size, mask=None):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    if mask is not None:
        h = w = int(mask.shape[1] ** .5)
        recon = torch.zeros(x.shape[0], h*w, x.shape[-1]).to(x)
        recon[mask] = x.flatten(0, 1)
    else:
        h = w = int(x.shape[1] ** .5)
        recon = x

    p = patch_size
    assert h * w == recon.shape[1]

    recon = recon.reshape(shape=(recon.shape[0], h, w, p, p, 3))
    recon = torch.einsum('nhwpqc->nchpwq', recon)
    imgs = recon.reshape(shape=(recon.shape[0], 3, h * p, h * p))
    return imgs


def sample_embedding(embedding, pos, mode='bilinear'):
    """
    Sample embedding tensor at specified positions
    embedding: [B, H, W, C]
    pos: [B, P, 2] (convention: first dim is row, second dim is column)
    """
    embedding = embedding.permute(0, 3, 1, 2) # [B, C, H, W]
    device = embedding.device
    # grid_sampling assues first value to be column-dimension, second value to be row-dimension
    pos = pos.flip(dims=(-1,))
    assert pos.min() >= -1 and pos.max() <= 1, "grid sampling expect to be in range [-1, 1]"

    return F.grid_sample(embedding, pos[:, None].to(device), mode=mode).squeeze(-2).permute(0, 2, 1)  # [B, P, C]


def sample_positions_from_dist(size, dist):
    """
    Samples positions from a given unnormalized probability distribution.

    Parameters:
    num (int): The number of samples to draw for each distribution in the batch.
    dist (torch.Tensor): A float tensor of shape [B, H, W] representing the unnormalized
                         probability distributions for B batches each of length N.

    Returns:
    torch.Tensor: A tensor of shape [B, num] containing the sampled positions.
    """
    assert dist.dim() == 3, "dist should be a 3D tensor with shape [B, H, W]."
    assert len(size) == 2, "size should be a 2D tuple (batch_size, num_samples)"
    B, H, W = dist.shape

    new_B, num_samples = size

    if dist.min() < 0:
        dist -= dist.min()

    # Flatten the last two dimensions to make it [B, H*W]
    flattened_dist = dist.view(B, -1)

    # Sample indices according to the normalized distribution
    sampled_indices = torch.multinomial(flattened_dist, new_B * num_samples, replacement=True)

    # Convert the flattened indices back to 2D indices
    sampled_row_indices = sampled_indices // W
    sampled_col_indices = sampled_indices % W

    # Stack the row and column indices
    samples = torch.stack((sampled_row_indices, sampled_col_indices), dim=-1)
    samples = samples.view(new_B, num_samples, 2)

    return samples

def get_dino_predominance(images, dims=[28, 28], current_mask=None, painting=None, img_size=[224, 224]):
    global dino_backbone
    if dino_backbone is None:
        with contextlib.redirect_stdout(io.StringIO()):
            vit_arch = 'base'
            vit_feat = 'k'
            patch_size = 8
            # DINO pre-trained model
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            feat_dim = 768
            dino_backbone = dino.ViTFeat(url, feat_dim, vit_arch, vit_feat, patch_size)
            dino_backbone = dino_backbone.eval().requires_grad_(False).cuda()

    input_dino = images
    input_dino = torch.nn.functional.interpolate(input_dino, size=img_size, mode='bilinear')
    features = dino_backbone(input_dino)

    predominence_map = []

    for i in range(features.shape[0]):
        feats = features[i]
        if current_mask == None:
            painting = torch.from_numpy(np.zeros(dims))
            painting = painting.to(feats)
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, ps=dims[0])

        A, D = get_affinity_matrix(feats, tau=0.15)
        # get the second-smallest eigenvector
        _, second_smallest_vec = second_smallest_eigenvector(A, D)
        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)

        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dims)
        if nc >= 2:
            reverse = True
        else:
            reverse = bipartition[seed] != 1
        if reverse:
            second_smallest_vec = 1 - second_smallest_vec
        second_smallest_vec = torch.tensor(second_smallest_vec).to(images.device).contiguous()
        map = torch.nn.functional.interpolate(second_smallest_vec.reshape(1, 1, dims[0], dims[1]), size=img_size,
                                              mode='bilinear')
        map -= map.min()
        map /= map.max()
        predominence_map.append(map)
    init_dist = torch.cat(predominence_map, dim=0).detach()
    return init_dist, A, feats, painting


def interpolate_pos_encoding(pos_embed, n_frames, h, w):
    N = pos_embed.shape[1]
    if N == (h * w * n_frames):
        return pos_embed
    old_h = old_w = int((N / n_frames) ** 0.5)
    patch_pos_embed = pos_embed.view(1, n_frames, old_h, old_w, -1).flatten(0, 1).permute(0, 3, 1, 2)

    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(h, w),
        mode='bicubic',
    )
    return patch_pos_embed.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(0)


def flow_to_rgb(vec, flow_mag_range=None, white_bg=False):
    height, width = vec.shape[:2]
    scaling = 50. / (height ** 2 + width ** 2) ** 0.5
    direction = (np.arctan2(vec[..., 0], vec[..., 1]) + np.pi) / (2 * np.pi)
    norm = np.linalg.norm(vec, axis=-1)
    if flow_mag_range is None:
        flow_mag_range = norm.min(), norm.max()
        magnitude = np.clip((norm - flow_mag_range[0]) * scaling, 0., 1.)
    if white_bg == True:
        value = np.ones_like(direction)
        hsv = np.stack([direction, magnitude, saturation], axis=-1)
    else:
        saturation = np.ones_like(direction)
        hsv = np.stack([direction, saturation, magnitude], axis=-1)
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb


def normalize_flow_mag(flow_mag, eps=1e-6):
    flow_mag -= flow_mag.amin((-2, -1), True)
    flow_mag /= flow_mag.amax((-2, -1), True).clamp(min=eps)
    return flow_mag


def compute_flow_mag_overlap(flow_mag_a, flow_mag_b, normalized=True, eps=1e-6):
    """
    flow_mag_a, flow_b are flow maps with shape [B, H, W]
    return [B,] representing the iou between the two maps
    """
    if normalized:
        flow_mag_a = normalize_flow_mag(flow_mag_a, eps)
        flow_mag_b = normalize_flow_mag(flow_mag_b, eps)

    inter = torch.minimum(flow_mag_a, flow_mag_b)
    union = torch.maximum(flow_mag_a, flow_mag_b)
    return inter.sum((-2, -1)) / union.sum((-2, -1)).clamp(min=eps)


def compute_pairwise_flow_mag_overlap(flow_mag_a, flow_mag_b, normalized=True, eps=1e-6):
    """
    Computes pairwise IoU between two sets of flow magnitude maps.

    Args:
    - flow_mag_a: Tensor of shape [B, N, H, W]
    - flow_mag_b: Tensor of shape [B, N, H, W]
    - normalized: If True, normalizes the flow magnitude maps
    - eps: A small value to avoid division by zero

    Returns:
    - Tensor of shape [B, N, N] representing the pairwise IoU between the maps
    """

    # Normalize flow magnitude maps if required

    if normalized:
        flow_mag_a = normalize_flow_mag(flow_mag_a, eps)
        flow_mag_b = normalize_flow_mag(flow_mag_b, eps)

    # Add dimensions to prepare for pairwise comparison
    # flow_mag_a: [B, N, H, W] -> [B, N, 1, H, W]
    # flow_mag_b: [B, N, H, W] -> [B, 1, N, H, W]
    flow_mag_a = flow_mag_a.unsqueeze(2)  # Shape: [B, N, 1, H, W]
    flow_mag_b = flow_mag_b.unsqueeze(1)  # Shape: [B, 1, N, H, W]

    # Compute element-wise minimum (intersection) and maximum (union)
    inter = torch.minimum(flow_mag_a, flow_mag_b)  # Shape: [B, N, N, H, W]
    union = torch.maximum(flow_mag_a, flow_mag_b)  # Shape: [B, N, N, H, W]

    # Sum over the height and width dimensions to get total intersection and union
    inter_sum = inter.sum(dim=(-2, -1))  # Shape: [B, N, N]
    union_sum = union.sum(dim=(-2, -1)).clamp(min=eps)  # Shape: [B, N, N], avoid division by zero with clamp

    # Compute pairwise IoU
    pairwise_iou = inter_sum / union_sum  # Shape: [B, N, N]

    # Create a mask to exclude diagonal elements
    N = pairwise_iou.shape[1]
    mask = torch.ones(N, N, dtype=bool)  # Shape: [N, N]
    mask.fill_diagonal_(False)  # Set diagonal to False

    # Apply mask to exclude diagonal elements and compute the mean for each batch
    pairwise_iou_non_diag = pairwise_iou[:, mask].view(pairwise_iou.shape[0], -1)  # Shape: [B, N*(N-1)]
    mean_non_diag_iou = pairwise_iou_non_diag.mean(dim=-1)  # Mean over non-diagonal elements for each batch

    return mean_non_diag_iou

def overlay_masks_on_image(image, segments, contour_thickness=1, seed=42):
    if len(segments) == 0:
        return image

    # Seed the random number generator to ensure consistent colors
    random.seed(seed)
    np.random.seed(seed)

    # Create an RGBA image for the mask overlay
    img = np.array(image.convert("RGBA"))
    overlay = np.zeros_like(img, dtype=np.float32)

    for m in segments:
        print('m.shape', m.shape, overlay.shape)
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # Random color with alpha
        overlay[m] = color_mask

        # Find contours
        if contour_thickness is not None and contour_thickness > 0:
            contours = measure.find_contours(m.numpy(), 0.5)
            for contour in contours:
                for dx in range(-contour_thickness, contour_thickness + 1):
                    for dy in range(-contour_thickness, contour_thickness + 1):
                        for point in contour:
                            y, x = point
                            ny, nx = int(y) + dy, int(x) + dx
                            if 0 <= ny < overlay.shape[0] and 0 <= nx < overlay.shape[1]:
                                overlay[ny, nx, :3] = color_mask[:3]
                                overlay[ny, nx, 3] = 1.0  # Fully opaque for the contour

    # Combine the original image with the overlay
    combined_img = img * (1 - overlay[..., 3:]) + overlay * 255
    combined_img = np.clip(combined_img, 0, 255).astype(np.uint8)

    # Convert back to PIL image
    result_image = Image.fromarray(combined_img)
    return result_image