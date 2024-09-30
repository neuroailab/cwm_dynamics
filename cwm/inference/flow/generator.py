import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import cwm.inference.flow.masking_utils as masking
from tqdm import tqdm
from cwm.inference.flow.flow_utils import imagenet_normalize, coordinate_ims, get_distribution_centroid


class Patchify(nn.Module):
    """Convert a set of images or a movie into patch vectors"""
    def __init__(self,
                 patch_size=(16, 16),
                 temporal_dim=1,
                 squeeze_channel_dim=True
                 ):
        super().__init__()
        self.set_patch_size(patch_size)
        self.temporal_dim = temporal_dim
        assert self.temporal_dim in [1, 2], self.temporal_dim
        self._squeeze_channel_dim = squeeze_channel_dim

    @property
    def num_patches(self):
        if (self.T is None) or (self.H is None) or (self.W is None):
            return None
        else:
            return (self.T // self.pt) * (self.H // self.ph) * (self.W // self.pw)

    def set_patch_size(self, patch_size):
        self.patch_size = patch_size
        if len(self.patch_size) == 2:
            self.ph, self.pw = self.patch_size
            self.pt = 1
            self._patches_are_3d = False
        elif len(self.patch_size) == 3:
            self.pt, self.ph, self.pw = self.patch_size
            self._patches_are_3d = True
        else:
            raise ValueError("patch_size must be a 2- or 3-tuple, but is %s" % self.patch_size)

        self.shape_inp = self.rank_inp = self.H = self.W = self.T = None
        self.D = self.C = self.E = self.embed_dim = None

    def _check_shape(self, x):
        self.shape_inp = x.shape
        self.rank_inp = len(self.shape_inp)
        self.H, self.W = self.shape_inp[-2:]
        assert (self.H % self.ph) == 0 and (self.W % self.pw) == 0, (self.shape_inp, self.patch_size)
        if (self.rank_inp == 5) and self._patches_are_3d:
            self.T = self.shape_inp[self.temporal_dim]
            assert (self.T % self.pt) == 0, (self.T, self.pt)
        elif self.rank_inp == 5:
            self.T = self.shape_inp[self.temporal_dim]
        else:
            self.T = 1

    def split_by_time(self, x):
        shape = x.shape
        assert shape[1] % self.T == 0, (shape, self.T)
        return x.view(shape[0], self.T, shape[1] // self.T, *shape[2:])

    def merge_by_time(self, x):
        shape = x.shape
        return x.view(shape[0], shape[1] * shape[2], *shape[3:])

    def video_to_patches(self, x):
        if self.rank_inp == 4:
            assert self.pt == 1, (self.pt, x.shape)
            x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw) c', ph=self.ph, pw=self.pw)
        else:
            assert self.rank_inp == 5, (x.shape, self.rank_inp, self.shape_inp)
            dim_order = 'b (t pt) c (h ph) (w pw)' if self.temporal_dim == 1 else 'b c (t pt) (h ph) (w pw)'
            x = rearrange(x, dim_order + ' -> b (t h w) (pt ph pw) c', pt=self.pt, ph=self.ph, pw=self.pw)

        self.N, self.D, self.C = x.shape[-3:]
        self.embed_dim = self.E = self.D * self.C
        return x

    def patches_to_video(self, x):
        shape = x.shape
        rank = len(shape)
        if rank == 4:
            B, _N, _D, _C = shape
        else:
            assert rank == 3, rank
            B, _N, _E = shape
            assert (_E % self.D == 0), (_E, self.D)
            x = x.view(B, _N, self.D, -1)

        if _N < self.num_patches:
            masked_patches = self.get_masked_patches(
                x,
                num_patches=(self.num_patches - _N),
                mask_mode=self.mask_mode)
            x = torch.cat([x, masked_patches], 1)

        x = rearrange(
            x,
            'b (t h w) (pt ph pw) c -> b c (t pt) (h ph) (w pw)',
            pt=self.pt, ph=self.ph, pw=self.pw,
            t=(self.T // self.pt), h=(self.H // self.ph), w=(self.W // self.pw))

        if self.rank_inp == 5 and (self.temporal_dim == 1):
            x = x.transpose(1, 2)
        elif self.rank_inp == 4:
            assert x.shape[2] == 1, x.shape
            x = x[:, :, 0]
        return x

    @staticmethod
    def get_masked_patches(x, num_patches, mask_mode='zeros'):
        shape = x.shape
        patches_shape = (shape[0], num_patches, *shape[2:])
        if mask_mode == 'zeros':
            return torch.zeros(patches_shape).to(x.device).to(x.dtype).detach()
        elif mask_mode == 'gray':
            return 0.5 * torch.ones(patches_shape).to(x.device).to(x.dtype).detach()
        else:
            raise NotImplementedError("Haven't implemented mask_mode == %s" % mask_mode)

    def average_within_patches(self, z):
        if len(z.shape) == 3:
            z = rearrange(z, 'b n (d c) -> b n d c', c=self.C)
        return z.mean(-2, True).repeat(1, 1, z.shape[-2], 1)

    def forward(self, x, to_video=False, mask_mode='zeros'):
        if not to_video:
            self._check_shape(x)
            x = self.video_to_patches(x)
            return x if not self._squeeze_channel_dim else x.view(x.size(0), self.N, -1)

        else:  # x are patches
            assert (self.shape_inp is not None) and (self.num_patches is not None)
            self.mask_mode = mask_mode
            x = self.patches_to_video(x)
            return x


class DerivativeFlowGenerator(nn.Module):
    """Estimate flow of a two-frame predictor using torch autograd"""

    def __init__(self,
                 predictor,
                 perturbation_patch_size=None,
                 aggregation_patch_size=None,
                 agg_power=None,
                 agg_channel_func=None,
                 num_samples=1,
                 leave_one_out_sampling=False,
                 average_jacobian=True,
                 confidence_thresh=None,
                 temporal_dim=2,
                 imagenet_normalize_inputs=True):

        super(DerivativeFlowGenerator, self).__init__()

        self.predictor = predictor

        self.patchify = Patchify(self.patch_size, temporal_dim=1, squeeze_channel_dim=True)

        self.set_temporal_dim(temporal_dim)

        self.imagenet_normalize_inputs = imagenet_normalize_inputs

        self.perturbation_patch_size = self._get_patch_size(perturbation_patch_size) or self.patch_size
        self.aggregation_patch_size = self._get_patch_size(aggregation_patch_size) or self.patch_size
        self.agg_patchify = Patchify(self.aggregation_patch_size,
                                     temporal_dim=1,
                                     squeeze_channel_dim=False)
        self.agg_channel_func = agg_channel_func or (lambda x: F.relu(x).sum(-3, True))
        self.average_jacobian = average_jacobian
        self.confidence_thresh = confidence_thresh

        self.num_samples = num_samples
        self.leave_one_out_sampling = leave_one_out_sampling
        self.agg_power = agg_power
        self.t_dim = temporal_dim

    def _get_patch_size(self, p):
        if p is None:
            return None
        elif isinstance(p, int):
            return (1, p, p)
        elif len(p) == 2:
            return (1, p[0], p[1])
        else:
            assert len(p) == 3, p
            return (p[0], p[1], p[2])

    def set_temporal_dim(self, t_dim):
        if t_dim == 1:
            self.predictor.t_dim = 1
            self.predictor.c_dim = 2
        elif t_dim == 2:
            self.predictor.c_dim = 1
            self.predictor.t_dim = 2
        else:
            raise ValueError("temporal_dim must be 1 or 2")

    @property
    def c_dim(self):
        if self.predictor is None:
            return None
        return self.predictor.c_dim

    @property
    def patch_size(self):
        if self.predictor is None:
            return None
        elif hasattr(self.predictor, 'patch_size'):
            return self.predictor.patch_size
        elif hasattr(self.predictor.encoder.patch_embed, 'proj'):
            return self.predictor.encoder.patch_embed.proj.kernel_size
        else:
            return None

    @property
    def S(self):
        return self.num_samples

    @property
    def sequence_length(self):
        if self.predictor is None:
            return None
        elif hasattr(self.predictor, 'sequence_length'):
            return self.predictor.sequence_length
        elif hasattr(self.predictor, 'num_frames'):
            return self.predictor.num_frames
        else:
            return 2

    @property
    def mask_shape(self):
        if self.predictor is None:
            return None
        elif hasattr(self.predictor, 'mask_shape'):
            return self.predictor.mask_shape

        assert self.patch_size is not None
        pt, ph, pw = self.patch_size
        return (self.sequence_length // pt,
                self.inp_shape[-2] // ph,
                self.inp_shape[-1] // pw)

    @property
    def perturbation_mask_shape(self):
        return (
            self.mask_shape[0],
            self.inp_shape[-2] // self.perturbation_patch_size[-2],
            self.inp_shape[-1] // self.perturbation_patch_size[-1]
        )

    @property
    def p_mask_shape(self):
        return self.perturbation_mask_shape

    @property
    def aggregation_mask_shape(self):
        return (
            1,
            self.inp_shape[-2] // self.aggregation_patch_size[-2],
            self.inp_shape[-1] // self.aggregation_patch_size[-1]
        )

    @property
    def a_mask_shape(self):
        return self.aggregation_mask_shape

    def get_perturbation_input(self, x):
        self.set_input(x)
        y = torch.zeros((self.B, *self.p_mask_shape), dtype=x.dtype, device=x.device, requires_grad=True)
        y = y.unsqueeze(2).repeat(1, 1, x.shape[2], 1, 1)
        return y

    def pred_patches_to_video(self, y, x, mask):
        """input at visible positions, preds at masked positions"""
        B, C = y.shape[0], y.shape[-1]
        self.patchify._check_shape(x)
        self.patchify.D = np.prod(self.patch_size)
        x = self.patchify(x)
        y_out = torch.zeros_like(x)
        x_vis = x[~mask]

        y_out[~mask] = x_vis.view(-1, C)
        try:
            y_out[mask] = y.view(-1, C)
        except:
            y_out[mask] = y.reshape(-1, C)

        return self.patchify(y_out, to_video=True)

    def set_image_size(self, *args, **kwargs):
        assert self.predictor is not None, "Can't set the image size without a predictor"
        if hasattr(self.predictor, 'set_image_size'):
            self.predictor.set_image_size(*args, **kwargs)
        else:
            self.predictor.image_size = args[0]

    def predict(self, x=None, mask=None, forward_full=False):
        if x is None:
            x = self.x
        if mask is None:
            mask = self.generate_mask(x)

        self.set_image_size(x.shape[-2:])
        y = self.predictor(
            self._preprocess(x),
            mask if (x.size(0) == 1) else self.mask_rectangularizer(mask), forward_full=forward_full)

        y = self.pred_patches_to_video(y, x, mask=mask)

        frame = -1 % y.size(1)
        y = y[:, frame:frame + 1]

        return y

    def _get_perturbation_func(self, x=None, mask=None):

        if (x is not None):
            self.set_input(x, mask)

        def forward_mini_image(y):
            #repeat the perturbation image to match the image size
            y = y.repeat_interleave(self.perturbation_patch_size[-2], -2)
            y = y.repeat_interleave(self.perturbation_patch_size[-1], -1)

            #predict the image with the added perturbation
            x_pred = self.predict(self.x + y, self.mask)

            #aggregate responses
            x_pred = self.agg_patchify(x_pred).mean(-2).sum(-1).view(self.B, *self.a_mask_shape)
            return x_pred[self.targets]

        return forward_mini_image

    def _postprocess_jacobian(self, jac):
        _jac = torch.zeros((self.B, *self.a_mask_shape, *jac.shape[1:])).to(jac.device).to(jac.dtype)
        _jac[self.targets] = jac
        jac = self.agg_channel_func(_jac)
        assert jac.size(-3) == 1, jac.shape
        jac = jac.squeeze(-3)[..., 0, :, :]  # derivative w.r.t. first frame and agg channels
        jac = jac.view(self.B, self.a_mask_shape[-2], self.a_mask_shape[-1],
                       self.B, self.p_mask_shape[-2], self.p_mask_shape[-1])
        bs = torch.arange(0, self.B).long().to(jac.device)
        jac = jac[bs, :, :, bs, :, :]  # take diagonal
        return jac

    def _confident_jacobian(self, jac):
        if self.confidence_thresh is None:
            return torch.ones_like(jac[:, None, ..., 0, 0])
        conf = (jac.amax((-2, -1)) > self.confidence_thresh).float()[:, None]
        return conf

    def set_input(self, x, mask=None, timestamps=None):
        shape = x.shape
        if len(shape) == 4:
            x = x.unsqueeze(1)
        else:
            assert len(shape) == 5, \
                "Input must be a movie of shape [B,T,C,H,W]" + \
                "or a single frame of shape [B,C,H,W]"

        self.inp_shape = x.shape
        self.x = x
        self.B = self.inp_shape[0]
        self.T = self.inp_shape[1]
        self.C = self.inp_shape[2]
        if mask is not None:
            self.mask = mask

        if timestamps is not None:
            self.timestamps = timestamps

    def _preprocess(self, x):
        if self.imagenet_normalize_inputs:
            x = imagenet_normalize(x)
        if self.t_dim != 1:
            x = x.transpose(self.t_dim, self.c_dim)
        return x

    def _jacobian_to_flows(self, jac):
        if self.agg_power is None:
            jac = (jac == jac.amax((-2, -1), True)).float()
        else:
            jac = torch.pow(jac, self.agg_power)

        jac = jac.view(self.B * np.prod(self.a_mask_shape[-2:]), 1, 1, *self.p_mask_shape[-2:])
        centroids = get_distribution_centroid(jac, normalize=False).view(
            self.B, self.a_mask_shape[-2], self.a_mask_shape[-1], 2)
        rescale = [self.a_mask_shape[-2] / self.p_mask_shape[-2],
                   self.a_mask_shape[-1] / self.p_mask_shape[-1]]
        centroids = centroids * torch.tensor(rescale, device=centroids.device).view(1, 1, 1, 2)

        flows = centroids - \
                coordinate_ims(1, 0, self.a_mask_shape[-2:], normalize=False).to(jac.device)
        flows = flows.permute(0, 3, 1, 2)
        px_scale = torch.tensor(self.aggregation_patch_size[-2:]).float().to(flows.device).view(1, 2, 1, 1)
        flows *= px_scale

        return flows

    def set_targets(self, targets=None, frame=-1):
        frame = frame % self.mask_shape[0]
        if targets is None:
            targets = self.get_mask_image(self.mask)[:, frame:frame + 1]
        else:
            assert len(targets.shape) == 4, targets.shape
            targets = targets[:, frame:frame + 1]
        self.targets = ~masking.upsample_masks(~targets, self.a_mask_shape[-2:])

    def _get_mask_partition(self, mask):
        mask = self.get_mask_image(mask)
        mask_list = masking.partition_masks(
            mask[:, 1:], num_samples=self.S, leave_one_out=self.leave_one_out_sampling)
        return [torch.cat([mask[:, 0:1].view(m.size(0), -1), m], -1)
                for m in mask_list]

    def _compute_jacobian(self, y):
        perturbation_func = self._get_perturbation_func()
        jac = torch.autograd.functional.jacobian(
            perturbation_func,
            y,
            vectorize=False)
        jac = self._postprocess_jacobian(jac)
        return jac

    def _upsample_mask(self, mask):
        return masking.upsample_masks(
            mask.view(mask.size(0), -1, *self.mask_shape[-2:]).float(), self.inp_shape[-2:])

    def get_mask_image(self, mask, upsample=False, invert=False, shape=None):
        if shape is None:
            shape = self.mask_shape
        mask = mask.view(-1, *shape)
        if upsample:
            mask = self._upsample_mask(mask)
        if invert:
            mask = 1 - mask
        return mask

    def forward(self, x, mask, targets=None):
        self.set_input(x, mask)
        y = self.get_perturbation_input(x)
        mask_list = self._get_mask_partition(mask)

        jacobian, flows, confident = [], [], []
        for s, mask_sample in tqdm(enumerate(mask_list), total=len(mask_list), desc="Processing", ncols=100):

            self.set_input(x, mask_sample)
            self.set_targets(targets)

            # t1 = time.time()
            jac = self._compute_jacobian(y)
            conf_jac = masking.upsample_masks(self._confident_jacobian(jac), self.a_mask_shape[-2:])
            jacobian.append(jac)
            confident.append(conf_jac)
            if not self.average_jacobian:
                flow = self._jacobian_to_flows(jac) * self.targets * conf_jac * \
                       masking.upsample_masks(self.get_mask_image(self.mask)[:, 1:], self.a_mask_shape[-2:])
                flows.append(flow)
            # print(t2 - t1)

        jacobian = torch.stack(jacobian, -1)
        confident = torch.stack(confident, -1)
        valid = torch.stack([masking.upsample_masks(
            self.get_mask_image(m)[:, 1:], self.a_mask_shape[-2:]) for m in mask_list], -1)
        valid = valid * confident

        if self.average_jacobian:
            _valid = valid[:, 0].unsqueeze(-2).unsqueeze(-2)
            jac = (jacobian * _valid.float()).sum(-1) / _valid.float().sum(-1).clamp(min=1)
            flows = self._jacobian_to_flows(jac) * \
                    masking.upsample_masks(_valid[:, None, ..., 0, 0, :].amax(-1).bool(), self.a_mask_shape[-2:])
            if targets is not None:
                self.set_targets(targets)
                flows *= self.targets
        else:
            flows = torch.stack(flows, -1)
            flows = flows.sum(-1) / valid.float().sum(-1).clamp(min=1)

        valid = valid * (targets[:, -1:].unsqueeze(-1) if targets is not None else 1)

        return (jacobian, flows, valid)
