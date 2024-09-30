from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from cwm.data.masking_generator import RotatedTableMaskingGenerator
from cwm.model.model_utils import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
import cwm.inference.flow.masking_utils as masking
from cwm.inference.segmentation import segment_utils
import cwm.utils as utils
import cwm.inference.flow.generator as generator
import matplotlib.pyplot as plt
import torch.nn.functional as F
from cwm.inference.flow import flow_utils
import cwm.inference.keypoints.keypoint_utils as keypoint_utils

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def interpolate_pos_encoding(pos_embed, n_frames, h, w):
    N = pos_embed.shape[1]
    if N == (h * w * n_frames):
        return pos_embed
    old_h = old_w = int((N / n_frames) ** 0.5)
    patch_pos_embed = pos_embed.view(1, n_frames, old_h, old_w, -1).flatten(0, 1).permute(0, 3, 1, 2)

    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(h, w),
        mode='bilinear',
    )
    return patch_pos_embed.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(0)


class CWMEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=(16, 16), in_chans=3, num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,  num_frames=16, block_func=Block, k_bias=False, use_learnable_pos_emb=False, block_kwargs={}):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = (tubelet_size,) + patch_size
        self.pt, self.ph, self.pw = self.patch_size
        self.h = int(img_size / self.ph)
        self.w = int(img_size / self.pw)
        self.hw = self.h * self.w
        self.dims = [self.h, self.w]

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
            num_frames=num_frames
        )
        num_patches = self.patch_embed.num_patches

        self.num_patches = num_patches
        self.num_frames = num_frames

        if use_learnable_pos_emb:
            self.use_learnable_pos_emb = True
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            # sine-cosine positional embeddings
            self.use_learnable_pos_emb = False
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_func(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, **block_kwargs, k_bias=k_bias)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self):
        return self.pos_embed

    def forward_block(self, x, idx):
        return self.blocks[idx](x)

    def forward_features(self, x, mask, move_pos=None, static_pos=None, movement=None):

        x = embed = self.patch_embed(x)
        pos_embed = self._get_pos_embed().type_as(x).to(x.device).clone()

        if not self.use_learnable_pos_emb:
            pos_embed = pos_embed.detach()

        x = x + pos_embed
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        if move_pos is not None:
            h, w = self.h, self.w
            first_frame_emb = embed[:, :self.hw].view(B, h, w, C)  # [B, h, w, C]
            last_frame_pos_emb = pos_embed[:, -self.hw:].view(1, h, w, C).expand(B, -1, -1, -1)  # [B, h, w, C]
            denominator = torch.tensor([self.h, self.w]).view(1, 1, 2).to(x.device)

            new_pos = move_pos + movement  # [B, P, 2]
            move_pos = move_pos / denominator * 2 - 1
            new_pos = (new_pos / denominator).clamp(0, 1) * 2 - 1  # handle special case where new_pos is out of bounds
            static_pos = static_pos / denominator * 2 - 1

            moving_emb = utils.sample_embedding(first_frame_emb, move_pos, mode='nearest')  # [B, P, C]
            moving_pos_emb = utils.sample_embedding(last_frame_pos_emb, new_pos, mode='nearest')  # [B, P, C]

            static_emb = utils.sample_embedding(first_frame_emb, static_pos, mode='nearest')  # [B, P, C]
            static_pos_emb = utils.sample_embedding(last_frame_pos_emb, static_pos, mode='nearest')  # [B, P, C]

            x_vis = torch.cat([x_vis, moving_emb + moving_pos_emb, static_emb + static_pos_emb], dim=1)

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def _set_inputs(self, *args, **kwargs):
        pass

    def forward(self, x, mask, move_pos=None, static_pos=None, movement=None):

        self._set_inputs(x, mask)
        x = self.forward_features(x, mask, move_pos, static_pos, movement)
        x = self.head(x)
        return x


class CWMDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=(16, 16), num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, block_func=Block, block_kwargs={}, k_bias=False
                 ):
        super().__init__()

        self.num_classes = num_classes

        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_func(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, **block_kwargs, k_bias=k_bias)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_block(self, x, idx):
        return self.blocks[idx](x)

    def get_last_tokens(self, x, return_token_num):
        if return_token_num > 0:
            return self.head(self.norm(x[:, -return_token_num:]))
        elif return_token_num == 0:
            return self.head(self.norm(x))[:, x.size(1):]
        else:
            return self.head(self.norm(x))

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class CWM(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    default_input_kwargs = {'unnormalize': True}

    def __init__(self,
                 img_size=224,
                 patch_size=(16, 16),
                 encoder_func=CWMEncoder,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 encoder_block_func=Block,
                 encoder_block_kwargs={},
                 decoder_num_classes=None,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 decoder_block_func=Block,
                 decoder_block_kwargs={},
                 mlp_ratio=4.,
                 qkv_bias=False,
                 k_bias=False,
                 qk_scale=None,
                 num_frames=16,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 tubelet_size=2,
                 use_learnable_pos_emb=False,
                 **kwargs
                 ):
        super().__init__()

        self.tubelet_size = tubelet_size
        num_classes = 3 * tubelet_size * (
                    patch_size[0] * patch_size[1]) if decoder_num_classes is None else decoder_num_classes

        self.encoder = encoder_func(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
            block_func=encoder_block_func,
            block_kwargs=encoder_block_kwargs,
            use_learnable_pos_emb=use_learnable_pos_emb,
            k_bias=k_bias)

        self.decoder = CWMDecoder(
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            block_func=decoder_block_func,
            k_bias=k_bias,
            block_kwargs=decoder_block_kwargs)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=k_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=.02)

        if use_learnable_pos_emb:
            self.use_learnable_pos_emb = True
            self.pos_embed = nn.Parameter(torch.zeros(self.encoder.num_patches, decoder_embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.use_learnable_pos_emb = False
            self.pos_embed = get_sinusoid_encoding_table(self.encoder.num_patches, decoder_embed_dim)

        self.num_frames = num_frames
        self.num_patches = self.encoder.num_patches

        if self.num_frames is not None:
            self.num_patches_per_frame = self.num_patches // self.num_frames
        else:
            self.num_patches_per_frame = self.num_patches

        self.patch_size = self.encoder.patch_size

        if isinstance(img_size, int):
            self.image_size = (img_size, img_size)
        else:
            assert hasattr(img_size, '__len__'), img_size
            self.image_size = img_size

    @property
    def mask_size(self):
        return (self.num_frames // self.patch_size[0],
                self.image_size[-2] // self.patch_size[-2],
                self.image_size[-1] // self.patch_size[-1])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def adjust_input_resolution(self, H, W):
        if self.image_size == [H, W]:
            return
        patch_size = self.encoder.patch_size[-2:]
        self.image_size = [H, W]
        self.encoder.h = int(H / self.encoder.ph)
        self.encoder.w = int(W / self.encoder.pw)
        self.encoder.hw = self.encoder.h * self.encoder.w
        self.encoder.dims = [self.encoder.h, self.encoder.w]
        dims = [int(s / p) for s, p in zip(self.image_size, patch_size)]
        self.encoder.pos_embed = utils.interpolate_pos_encoding(self.encoder.pos_embed, 3, dims[0], dims[1])
        self.pos_embed = utils.interpolate_pos_encoding(self.pos_embed, 3, dims[0], dims[1])

    def forward(self, x, mask, forward_full=False, return_features=False, *args, **kwargs):

        _, _, T, _, _ = x.shape
        self.device = x.device

        num_patches_per_frame = (x.shape[-1] // self.encoder.patch_size[1]) ** 2

        x_vis = self.encoder(x, mask, *args, **kwargs)

        if return_features:
            return x_vis

        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape

        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone()

        if not self.use_learnable_pos_emb:
            expand_pos_embed = expand_pos_embed.detach()

        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)

        nctx = num_patches_per_frame * (self.num_frames - 1)

        x_vis = x_vis + pos_emd_vis

        x_full = torch.cat([x_vis, self.mask_token + pos_emd_mask], dim=1)  # [B, N, C_d]

        if forward_full:
            x_full = torch.cat([x_vis, self.mask_token + expand_pos_embed[:, nctx:]], dim=1)  # [B, N, C_d]
            x_all = self.decoder(x_full, num_patches_per_frame)
            x = x_all
        else:
            x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]

        return x

    def get_intervention_outcome(self, x, move_patches):
        '''
        :param x: input tensor [1, C, T, H, W]: support only batch size 1 for now
        :param move_patches: torch tensor [N, 4] sized array where each row contains patch motion [x1, y1, x2, y2] in pixel coordinates
        :return:
        '''
        B, _, T, H, H  = x.shape

        mask = torch.ones(B, self.encoder.hw * self.encoder.num_frames).to(x.device).bool()
        mask[:, :self.encoder.hw * (self.encoder.num_frames - 1)] = False

        move_patches = (move_patches/H)*self.encoder.h
        move_patches = move_patches.to(torch.int64)

        for x1, y1, x2, y2 in move_patches:
            idx2 = x2*self.encoder.w + y2 + (self.encoder.num_frames - 1) * (self.encoder.h * self.encoder.w)
            mask[:, idx2] = False
            im_x1 = x1*self.encoder.ph
            im_y1 = y1*self.encoder.pw
            im_x2 = x2*self.encoder.ph
            im_y2 = y2*self.encoder.pw
            x[:, :, -1, im_x2:im_x2+self.encoder.ph, im_y2:im_y2+self.encoder.pw] = x[:, :, -2, im_x1:im_x1+self.encoder.ph, im_y1:im_y1+self.encoder.pw]

        prediction = self.forward(x, mask, forward_full=True)

        prediction = utils.unpatchify_cwm(
            prediction,
            patch_size=self.encoder.patch_size[-1],
        )  # reshape the output to an image

        return prediction


    def get_directional_intervention_outcome(self, x, mask=None, move_pos=None, static_pos=None, movement=None, max_movement=None):
        B, _, T, _, _ = x.shape

        if mask is None:  # default mask: all visible but the last frame
            mask = torch.ones(B, self.encoder.hw * self.encoder.num_frames).to(x.device).bool()
            mask[:, :self.encoder.hw * (self.encoder.num_frames - 1)] = False

        if movement is None:  # generate random motion if movement is not specified
            assert max_movement is not None and move_pos is not None
            movement = torch.randint(-max_movement, max_movement, move_pos.shape).to(x.device)  # [B, num_samples, 2]

        x_vis = self.encoder(x, mask, move_pos=move_pos, static_pos=static_pos, movement=movement)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)

        if move_pos is not None:
            h, w = self.encoder.h, self.encoder.w
            last_frame_pos_emb = expand_pos_embed[:, -(h * w):].view(B, h, w, C)  # [B, h, w, C]

            # compute new locations of the moved patche, snormalize positions to range [-1, 1]
            new_pos = move_pos + movement  # [B, P, 2]
            denominator = torch.tensor([h, w]).view(1, 1, 2).to(x.device)
            new_pos = (new_pos / denominator).clamp(0, 1) * 2 - 1
            static_pos = static_pos / denominator * 2 - 1

            # sample the position embeddings of the moved and static patches
            moving_pos_emb = utils.sample_embedding(last_frame_pos_emb, new_pos, mode='nearest')  # [B, P, C]
            static_pos_emb = utils.sample_embedding(last_frame_pos_emb, static_pos, mode='nearest')  # [B, P, C]

            # concatenate with the position embeddings to the visible patches
            pos_emd_vis = torch.cat([pos_emd_vis, moving_pos_emb, static_pos_emb], dim=1)

        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_vis = x_vis + pos_emd_vis
        x_full = torch.cat([x_vis, self.mask_token + pos_emd_mask], dim=1)  # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]

        prediction = utils.unpatchify_cwm(
            x,
            patch_size=self.encoder.patch_size[-1],
            mask=mask[:, -self.encoder.hw:]
        )  # reshape the output to an image

        return prediction


    @torch.no_grad()
    def get_segment(self, img, pos, num_samples=16, num_static=2, max_movement=8):
        '''
        Extracts segments from an input image at specified pixel positions using counterfactual motion,
        and returns the segmentation of shape [B, H, W].

        :param img: Input image tensor of shape [B, C, H, W]
        :param pos: List of (x, y) pixel positions where segments will be generated.
        :param num_samples: Number of counterfactual motion samples to generate (default: 16).
        :param num_static: Number of static patches to sample from the background, which helps prevent
                           camera panning and focuses on object motion (default: 2).
        :param max_movement: Maximum magnitude of movement allowed in generating counterfactual motion.
                             (default: 8).
        :return: A segmentation map of shape [B, H, W], where B is 1, H is height, and W is width.
        '''
        assert img.shape[0] == 1, "current implentation is tested for batch size == 1"
        segments = segment_utils.get_segment_at_pos(
            self,
            img,
            pos=pos,
            num_samples=num_samples,
            num_static=num_static,
            max_movement=max_movement
        )

        return segments

    def get_flow_jacobian_method(self, img1, img2,
                 perturbation_patch_size=8,
                 aggregation_patch_size=8,
                 mask_ratio=0.0):
        '''
        :param img1: input image 1 [B, C, H, W]
        :param img2: input image 2 [B, C, H, W]
        :param perturbation_patch_size: size of the patch to perturb when computing the jacobian
        :param aggregation_patch_size: size of the patch over which to aggregate responses
        :return: forward flow [B, 2, H, W]
        '''

        predictor_image_size = self.image_size[-1]


        frame_size = predictor_image_size // self.patch_size[-1]
        DFG = generator.DerivativeFlowGenerator(
            predictor=self,
            perturbation_patch_size=perturbation_patch_size,
            aggregation_patch_size=aggregation_patch_size,
            agg_power=None,
            agg_channel_func=lambda x: F.relu(x.sum(-3, True)),
            num_samples=5,
            average_jacobian=False,
            leave_one_out_sampling=False,
            imagenet_normalize_inputs=False,
            temporal_dim=2,
            confidence_thresh=None
        ).to(img1.device)

        maskgen_uniform = masking.PytorchMaskGeneratorWrapper(
            mask_generator=masking.RotatedTableMaskingGenerator,
            input_size=(self.num_frames, frame_size, frame_size),
            mask_ratio=mask_ratio
        ).to(img1.device)

        jac_fwd, forward_flow = flow_utils.extract_jacobians_and_flows(img1, img2,
                                                                       DFG,
                                                                       maskgen_uniform()[None])

        return forward_flow

    @torch.no_grad()
    def get_flow_cost_volume_method(self, img1, img2,
                 conditioning_img=None,
                 mask_ratio=0.0,
                 num_scales=1,
                 num_mask_samples=1):
        '''
        :param img1: input image 1 [B, C, H, W]
        :param img2: input image 2 [B, C, H, W]
        :param mask_ratio: what frame2 mask ratio to use when extracting flow
        :param num_scales: number of scales at which to compute flow
        :param num_mask_samples: number of random samples of the target frame mask to use for getting a better statistical estimate of the optical flow
        :return: forward flow [B, 2, H, W]
        '''

        predictor_image_size = self.image_size[-1]

        frame_size = predictor_image_size // self.patch_size[-1]
        mask_generator = RotatedTableMaskingGenerator(
            input_size=(self.num_frames, frame_size, frame_size),
            mask_ratio=mask_ratio,
            tube_length=1,
            batch_size=1,
            mask_type='rotated_table'
        )

        forward_flow = flow_utils.extract_optical_flow(self, mask_generator, img1, img2, conditioning_img=conditioning_img, num_scales=num_scales, num_mask_samples=num_mask_samples)

        return forward_flow

    @torch.no_grad()
    def get_keypoints(self, img1, img2, img3=None, num_keypoints=10, samples_per_keypoint=1):
        '''
        :param img1: input image 1 [B, C, H, W] imagenet normalized
        :param img2: input image 2 [B, C, H, W] imagenet normalized
        :param img3: input image 3 [B, C, H, W] imagenet normalized. Is set to None for a 2 frame model
        :param num_keypoints: number of keypoints to extract
        :param samples_per_keypoint: number of samples per keypoint
        :return:
           mask: final mask with discovered keypoint patch locations set to True [B, number_of_patches]
           choices: x, y indices of keypoints [B, n_rounds, 2]
           err_array: the error maps for every keypoint found List [[B, n_samples, H, W]]*n_rounds
           feat: the features at the final state of the mask [B, C, H, W]
           keypoint_recon: the final reconstruction of the target frame (of the first element in the batch) given the keypoint patches [H, W, C]
        '''
        if self.num_frames == 2:
            x = torch.stack([img1, img2], dim=2)
        else:
            if img3 is None:
                x = torch.stack([img1, img1, img2], dim=2)
            else:
                x = torch.stack([img1, img2, img3], dim=2)

        mask, choices, err_array, feat, keypoint_recon = keypoint_utils.get_keypoints_batch(self, x, samples_per_keypoint, num_keypoints)

        return mask, choices, err_array, feat, keypoint_recon

def pretrain_vit_base_224_scaffold(img_size=224, **kwargs):
    model = CWM(
        img_size=img_size,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        decoder_depth=8,
        mlp_ratio=4,
        qkv_bias=True,
        k_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model

def pretrain_vit_large_224_scaffold(**kwargs):
    model = CWM(
        img_size=224,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        decoder_depth=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model

def pretrain_videomae_base_224_scaffold(**kwargs):
    model = CWM(
        img_size=224,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        decoder_depth=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model



def vitb_8x8patch_3frames(**kwargs):
    model = pretrain_vit_base_224_scaffold(
        patch_size=(8, 8),
        num_frames=3,
        tubelet_size=1,
        **kwargs)
    return model

def vitl_8x8patch_3frames(**kwargs):
    model = pretrain_vit_large_224_scaffold(
        patch_size=(8, 8),
        num_frames=3,
        tubelet_size=1,
        **kwargs)
    return model

def vitb_8x8patch_2frames(**kwargs):
    model = pretrain_vit_base_224_scaffold(
        patch_size=(8, 8),
        num_frames=2,
        tubelet_size=1,
        **kwargs)
    return model

def vitb_8x8patch_2frames_vmae(**kwargs):
    model = pretrain_videomae_base_224_scaffold(
        patch_size=(8, 8),
        num_frames=2,
        tubelet_size=1,
        **kwargs)
    return model

def vitb_4x4patch_2frames(**kwargs):
    model = pretrain_videomae_base_224_scaffold(
        patch_size=(4, 4),
        num_frames=2,
        tubelet_size=1,
        **kwargs)
    return model

from cwm.model.model_learnable_pos_embed import pretrain_vit_base_256_scaffold

def vitb_8x8patch_2frames_learnable_pos_embed(**kwargs):
    model = pretrain_vit_base_256_scaffold(
        patch_size=(8, 8),
        num_frames=2,
        tubelet_size=1,
        interp_noise=False,
        learn_pos_embed=True,
        **kwargs)
    return model

