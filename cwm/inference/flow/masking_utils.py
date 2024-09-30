import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def upsample_masks(masks, size, thresh=0.5):
    shape = masks.shape
    dtype = masks.dtype
    h, w = shape[-2:]
    H, W = size
    if (H == h) and (W == w):
        return masks
    elif (H < h) and (W < w):
        s = (h // H, w // W)
        return masks[..., ::s[0], ::s[1]]

    masks = masks.unsqueeze(-2).unsqueeze(-1)
    masks = masks.repeat(*([1] * (len(shape) - 2)), 1, H // h, 1, W // w)
    if ((H % h) == 0) and ((W % w) == 0):
        masks = masks.view(*shape[:-2], H, W)
    else:
        _H = np.prod(masks.shape[-4:-2])
        _W = np.prod(masks.shape[-2:])
        masks = transforms.Resize(size)(masks.view(-1, 1, _H, _W)) > thresh
        masks = masks.view(*shape[:2], H, W).to(masks.dtype)
    return masks




def partition_masks(masks, num_samples=2, leave_one_out=False):
    B = masks.shape[0]
    S = num_samples
    masks = masks.view(B, -1)
    partitioned = [torch.ones_like(masks) for _ in range(S)]
    for b in range(B):
        vis_inds = torch.where(~masks[b])[0]
        vis_inds = vis_inds[torch.randperm(vis_inds.size(0))]
        if leave_one_out:
            for s in range(S):
                partitioned[s][b][vis_inds] = 0
                partitioned[s][b][vis_inds[s::S]] = 1
        else:
            for s in range(S):
                partitioned[s][b][vis_inds[s::S]] = 0
    return partitioned


class RectangularizeMasks(nn.Module):
    """Make sure all masks in a batch have same number of 1s and 0s"""

    def __init__(self, truncation_mode='min'):
        super().__init__()
        self._mode = truncation_mode
        assert self._mode in ['min', 'max', 'mean', 'full', 'none', None], (self._mode)

    def set_mode(self, mode):
        self._mode = mode

    def __call__(self, masks):

        if self._mode in ['none', None]:
            return masks

        assert isinstance(masks, torch.Tensor), type(masks)
        if self._mode == 'full':
            return torch.ones_like(masks)

        shape = masks.shape
        masks = masks.flatten(1)
        B, N = masks.shape
        num_masked = masks.float().sum(-1)
        M = {
            'min': torch.amin, 'max': torch.amax, 'mean': torch.mean
        }[self._mode](num_masked).long()

        num_changes = num_masked.long() - M

        for b in range(B):
            nc = num_changes[b]
            if nc > 0:
                inds = torch.where(masks[b])[0]
                inds = inds[torch.randperm(inds.size(0))[:nc].to(inds.device)]
                masks[b, inds] = 0
            elif nc < 0:
                inds = torch.where(~masks[b])[0]
                inds = inds[torch.randperm(inds.size(0))[:-nc].to(inds.device)]
                masks[b, inds] = 1
        if list(masks.shape) != list(shape):
            masks = masks.view(*shape)

        return masks


class UniformMaskingGenerator(object):
    def __init__(self, input_size, mask_ratio, seed=None, clumping_factor=1, randomize_num_visible=False):
        self.frames = None
        if len(input_size) == 3:
            self.frames, self.height, self.width = input_size
        elif len(input_size) == 2:
            self.height, self.width = input_size
        elif len(input_size) == 1 or isinstance(input_size, int):
            self.height = self.width = input_size

        self.clumping_factor = clumping_factor
        self.pad_h = self.height % self.c[0]
        self.pad_w = self.width % self.c[1]
        self.num_patches_per_frame = (self.height // self.c[0]) * (self.width // self.c[1])
        self.mask_ratio = mask_ratio

        self.rng = np.random.RandomState(seed=seed)
        self.randomize_num_visible = randomize_num_visible

    @property
    def num_masks_per_frame(self):
        if not hasattr(self, '_num_masks_per_frame'):
            self._num_masks_per_frame = int(self.mask_ratio * self.num_patches_per_frame)
        return self._num_masks_per_frame

    @num_masks_per_frame.setter
    def num_masks_per_frame(self, val):
        self._num_masks_per_frame = val
        self._mask_ratio = (val / self.num_patches_per_frame)

    @property
    def c(self):
        if isinstance(self.clumping_factor, int):
            return (self.clumping_factor, self.clumping_factor)
        else:
            return self.clumping_factor[:2]

    @property
    def mask_ratio(self):
        return self._mask_ratio

    @mask_ratio.setter
    def mask_ratio(self, val):
        self._mask_ratio = val
        self._num_masks_per_frame = int(self._mask_ratio * self.num_patches_per_frame)

    @property
    def num_visible(self):
        return self.num_patches_per_frame - self.num_masks_per_frame

    @num_visible.setter
    def num_visible(self, val):
        self.num_masks_per_frame = self.num_patches_per_frame - val

    def __repr__(self):
        repr_str = "Mask: total patches per frame {}, mask patches per frame {}, mask ratio {}, random num num visible? {}".format(
            self.num_patches_per_frame, self.num_masks_per_frame, self.mask_ratio, self.randomize_num_visible
        )
        return repr_str

    def sample_mask_per_frame(self):
        num_masks = self.num_masks_per_frame
        if self.randomize_num_visible:
            num_masks = self.rng.randint(low=num_masks, high=(self.num_patches_per_frame + 1))
        mask = np.hstack([
            np.zeros(self.num_patches_per_frame - num_masks),
            np.ones(num_masks)])
        self.rng.shuffle(mask)
        if max(*self.c) > 1:
            mask = mask.reshape(self.height // self.c[0],
                                1,
                                self.width // self.c[1],
                                1)
            mask = np.tile(mask, (1, self.c[0], 1, self.c[1]))
            mask = mask.reshape((self.height - self.pad_h, self.width - self.pad_w))
            _pad_h = self.rng.choice(range(self.pad_h + 1))
            pad_h = (self.pad_h - _pad_h, _pad_h)
            _pad_w = self.rng.choice(range(self.pad_w + 1))
            pad_w = (self.pad_w - _pad_w, _pad_w)
            mask = np.pad(mask,
                          (pad_h, pad_w),
                          constant_values=1
                          ).reshape((self.height, self.width))
        return mask

    def __call__(self, num_frames=None):
        num_frames = (num_frames or self.frames) or 1
        masks = np.stack([self.sample_mask_per_frame() for _ in range(num_frames)]).flatten()
        return masks


class TubeMaskingGenerator(UniformMaskingGenerator):

    def __call__(self, num_frames=None):
        num_frames = (num_frames or self.frames) or 1
        masks = np.tile(self.sample_mask_per_frame(), (num_frames, 1)).flatten()
        return masks


class RotatedTableMaskingGenerator(TubeMaskingGenerator):

    def __init__(self, tube_length=None, *args, **kwargs):
        super(RotatedTableMaskingGenerator, self).__init__(*args, **kwargs)
        self.tube_length = tube_length

    def __call__(self, num_frames=None):
        num_frames = (num_frames or self.frames) or 2
        tube_length = self.tube_length or (num_frames - 1)
        table_thickness = num_frames - tube_length
        assert tube_length < num_frames, (tube_length, num_frames)

        tubes = super().__call__(num_frames=tube_length)
        top = np.zeros(table_thickness * self.height * self.width).astype(tubes.dtype).flatten()
        masks = np.concatenate([top, tubes], 0)
        return masks


class PytorchMaskGeneratorWrapper(nn.Module):
    """Pytorch wrapper for numpy masking generators"""

    def __init__(self,
                 mask_generator=TubeMaskingGenerator,
                 *args, **kwargs):
        super().__init__()
        self.mask_generator = mask_generator(*args, **kwargs)

    @property
    def mask_ratio(self):
        return self.mask_generator.mask_ratio

    @mask_ratio.setter
    def mask_ratio(self, value):
        self.mask_generator.mask_ratio = value

    def forward(self, device='cuda', dtype_out=torch.bool, **kwargs):
        masks = self.mask_generator(**kwargs)
        masks = torch.tensor(masks).to(device).to(dtype_out)
        return masks


class MaskingGenerator(nn.Module):
    """Pytorch base class for masking generators"""

    def __init__(self,
                 input_size,
                 mask_ratio,
                 seed=0,
                 visible_frames=0,
                 clumping_factor=1,
                 randomize_num_visible=False,
                 create_on_cpu=True,
                 always_batch=False):
        super().__init__()
        self.frames = None

        if len(input_size) == 3:
            self.frames, self.height, self.width = input_size
        elif len(input_size) == 2:
            self.height, self.width = input_size
        elif len(input_size) == 1 or isinstance(input_size, int):
            self.height = self.width = input_size

        self.clumping_factor = clumping_factor
        self.pad_h = self.height % self.c[0]
        self.pad_w = self.width % self.c[1]
        self.num_patches_per_frame = (self.height // self.c[0]) * (self.width // self.c[1])

        self.mask_ratio = mask_ratio
        self.visible_frames = visible_frames
        self.always_batch = always_batch
        self.create_on_cpu = create_on_cpu

        self.rng = np.random.RandomState(seed=seed)
        self._set_torch_seed(seed)

        self.randomize_num_visible = randomize_num_visible

    @property
    def num_masks_per_frame(self):
        if not hasattr(self, '_num_masks_per_frame'):
            self._num_masks_per_frame = int(self.mask_ratio * self.num_patches_per_frame)
        return self._num_masks_per_frame

    @num_masks_per_frame.setter
    def num_masks_per_frame(self, val):
        self._num_masks_per_frame = val
        self._mask_ratio = (val / self.num_patches_per_frame)

    @property
    def c(self):
        if isinstance(self.clumping_factor, int):
            return (self.clumping_factor,) * 2
        else:
            return self.clumping_factor[:2]

    @property
    def mask_ratio(self):
        return self._mask_ratio

    @mask_ratio.setter
    def mask_ratio(self, val):
        self._mask_ratio = val
        self._num_masks_per_frame = int(self._mask_ratio * self.num_patches_per_frame)

    @property
    def num_visible(self):
        return self.num_patches_per_frame - self.num_masks_per_frame

    @num_visible.setter
    def num_visible(self, val):
        self.num_masks_per_frame = self.num_patches_per_frame - val

    def _set_torch_seed(self, seed):
        self.seed = seed
        torch.manual_seed(self.seed)

    def __repr__(self):
        repr_str = ("Class: {}\nMask: total patches per mask {},\n" + \
                    "mask patches per mask {}, visible patches per mask {}, mask ratio {:0.3f}\n" + \
                    "randomize num visible? {}").format(
            type(self).__name__, self.num_patches_per_frame,
            self.num_masks_per_frame, self.num_visible, self.mask_ratio,
            self.randomize_num_visible
        )
        return repr_str

    def sample_mask_per_frame(self, *args, **kwargs):
        num_masks = self.num_masks_per_frame
        if self.randomize_num_visible:
            num_masks = self.rng.randint(low=num_masks, high=(self.num_patches_per_frame + 1))

        mask = torch.cat([
            torch.zeros([self.num_patches_per_frame - num_masks]),
            torch.ones([num_masks])], 0).bool()
        inds = torch.randperm(mask.size(0)).long()
        mask = mask[inds]

        if max(*self.c) > 1:
            mask = mask.view(self.height // self.c[0],
                             1,
                             self.width // self.c[1],
                             1)
            mask = torch.tile(mask, (1, self.c[0], 1, self.c[1]))
            mask = mask.reshape(self.height - self.pad_h, self.width - self.pad_w)
            _pad_h = self.rng.choice(range(self.pad_h + 1))
            pad_h = (self.pad_h - _pad_h, _pad_h)
            _pad_w = self.rng.choice(range(self.pad_w + 1))
            pad_w = (self.pad_w - _pad_w, _pad_w)
            mask = F.pad(mask,
                         pad_w + pad_h,
                         mode='constant',
                         value=1)
            mask = mask.reshape(self.height, self.width)

        return mask

    def forward(self, x=None, num_frames=None):

        num_frames = (num_frames or self.frames) or 1
        if isinstance(x, torch.Tensor):
            batch_size = x.size(0)
            masks = torch.stack([
                torch.cat([self.sample_mask_per_frame() for _ in range(num_frames)], 0).flatten()
                for b in range(batch_size)], 0)
            if not self.create_on_cpu:
                masks = masks.to(x.device)
            if batch_size == 1 and not self.always_batch:
                masks = masks.squeeze(0)
        else:
            batch_size = 1
            masks = torch.cat([self.sample_mask_per_frame() for _ in range(num_frames)], 0).flatten()
            if self.always_batch:
                masks = masks[None]

        if self.visible_frames > 0:
            vis = torch.zeros((batch_size, 1, self.height, self.width), dtype=torch.bool)
            vis = vis.view(masks.shape).to(masks.device)
            masks = torch.cat(([vis] * self.visible_frames) + [masks], -1)

        return masks
