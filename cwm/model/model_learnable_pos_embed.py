from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange
from cwm.model.model_utils import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table

from torch import Tensor
import cwm.utils as utils

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
        mode='bicubic',
    )
    return patch_pos_embed.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(0)

PRINT_PADDING = False

class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=(16, 16), in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False, num_frames=16, embed_per_frame=False, clumping_factor=None, block_func=Block, k_bias=False, interp_noise=False, block_kwargs={},learn_pos_embed=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = (tubelet_size,) + patch_size
        self.pt, self.ph, self.pw = self.patch_size
        self.h = int(img_size / self.ph)
        self.w = int(img_size / self.pw)
        self.hw = self.h * self.w

        self.clumping_factor = clumping_factor
        self.interp_noise = interp_noise

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if self.clumping_factor is not None:  # Clump the context frame for memory efficiency
            self.clumping_embed = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim,
                                            kernel_size=(tubelet_size, clumping_factor, clumping_factor),
                                            stride=(tubelet_size, clumping_factor, clumping_factor))

        self._embed_per_frame = embed_per_frame
        if not self._embed_per_frame:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size,num_frames=num_frames)
            num_patches = self.patch_embed.num_patches
        elif self._embed_per_frame:
            assert (num_frames % tubelet_size) == 0
            num_embeddings = (num_frames // tubelet_size)
            self.patch_embed = nn.ModuleList([
                PatchEmbed(
                    img_size=img_size, patch_size=patch_size,
                    in_chans=in_chans, embed_dim=embed_dim,
                    tubelet_size=tubelet_size, num_frames=tubelet_size)
                for _ in range(num_embeddings)])
            num_patches = self.patch_embed[0].num_patches * num_embeddings

        self.num_patches = num_patches
        self.num_frames = num_frames
        print("NUM PATCHES IN ENCODER", self.num_patches)

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        if learn_pos_embed:
            self.pos_embed = nn.Parameter(self.pos_embed)

        self.learn_pos_embed = learn_pos_embed

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_func(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, **block_kwargs, k_bias=k_bias)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _set_pos_embed(self, dim=None):
        if dim is None:
            dim = self.embed_dim
        if self.pos_embed is None:
            self.pos_embed = get_sinusoid_encoding_table(
                self.num_patches, dim)


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

    def interpolate_tensor_with_mask_token(self,
            x: Tensor, mask: Tensor, mask_token: Tensor, invert: bool = True
    ) -> Tensor:
        """
        Where mask == (0 if invert else 1), return x
        where mask == (1 if invert else 0), return mask_token
        Linearly interpolate between these using value of mask.
        """

        B, N, C = x.shape
        assert mask.shape[1] == N, (
            f"Number of tokens in mask ({mask.shape[1]}) does not match "
            f"number of tokens in input ({N})"
        )

        assert mask_token.shape[-1] == C, (
            f"Dimensionality of mask token ({mask_token.shape[-1]}) does not match "
            f"dimensionality of tokens in input ({C})"
        )

        # convert mask to interpolation weights in range [0., 1.]
        mask = mask.to(x).clip(min=0.0, max=1.0)
        mask = (1.0 - mask) if invert else mask
        mask = mask.unsqueeze(-1)  # [B, N, 1]

        # expand mask token
        mask_token = mask_token.view(1, 1, C).expand(B, N, -1)

        # interpolate
        start = mask_token
        end = x

        return start + mask * (end - start)

    def interpolate_tensor_with_noise(self,
            x: Tensor, mask: Tensor, invert: bool = True
    ) -> Tensor:
        """
        Where mask == (0 if invert else 1), return x
        where mask == (1 if invert else 0), return mask_token
        Linearly interpolate between these using value of mask.
        """
        # mask_token = mask_token
        # breakpoint()
        B, N, C = x.shape
        assert mask.shape[1] == N, (
            f"Number of tokens in mask ({mask.shape[1]}) does not match "
            f"number of tokens in input ({N})"
        )

        # convert mask to interpolation weights in range [0., 1.]
        mask = mask.to(x).clip(min=0.0, max=1.0)
        mask = (1.0 - mask) if invert else mask
        mask = mask.unsqueeze(-1)  # [B, N, 1]

        # ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Generate a 3x8x8 patch of random numbers from a normal distribution
        # with the same mean and std as ImageNet images
        rand_vec = torch.randn(B, N, 3, self.patch_size[-2], self.patch_size[-1]) * std + mean

        rand_vec = rand_vec.to(x.device).to(x.dtype).view(B, N, -1)
        # interpolate
        start = rand_vec
        end = x

        return start + mask * (end - start)

    def tokenize(self, x, mask=None):

        if not self._embed_per_frame:
            x = self.patch_embed(x)
        elif self._embed_per_frame:
            x = torch.cat([
                self.patch_embed[i](
                    x[:,:,(i*self.pt):((i+1)*self.pt)])
                for i in range(len(self.patch_embed))], 1)
            
        pos_embed = self._get_pos_embed().type_as(x).to(x.device).clone()
        if not self._learnable_pos_embed:
            pos_embed = pos_embed.detach()
        x = x + pos_embed
        return (x, mask)

    def tokenize_and_mask(self, x, mask):

        x, mask = self.tokenize(x, mask)
        B, _, C = x.shape
        # breakpoint()
        x_vis = x[~mask].reshape(B, -1, C)
        return x_vis

    def tokenize_and_mask_variable_size(self, x, mask):

        x, mask = self.tokenize(x, mask)
        B, _, C = x.shape
        all_batches = []
        max_len = 0
        all_len = []
        for i in range(B):
            x_vis = x[i, ~mask[i]]
            if x_vis.shape[0] > max_len:
                max_len = x_vis.shape[0]
            all_batches.append(x_vis)
            all_len.append(x_vis.shape[0])

        #pad all batches to max_len in a single line
        x_vis = torch.stack([F.pad(batch, (0,0,0,max_len-batch.shape[0]), mode='constant', value=0) for batch in all_batches])

        return x_vis, all_len

    def forward_features(self, x, mask, move_patches, static_patches, delta, mask_token, res=1, return_feat_layer=None):
        _, _, T, H, W = x.shape

        if self.interp_noise:
            #patchify x with patch size[0], patch size[1]
            p0 = self.patch_size[-2]
            p1 = self.patch_size[-1]
            x = rearrange(x, 'b c t (h p0) (w p1) -> b (t h w) (p0 p1 c)', p0=p0, p1=p1, h=H//p0, w=W//p1) # x: [B, N, C]

            x = self.interpolate_tensor_with_noise(x, mask, invert=True)
            x = rearrange(x, 'b n (p c) -> b n p c', c=3)
            # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
            x = rearrange(x,
                            'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
                            p0=1,
                            p1=self.patch_size[-2],
                            p2=self.patch_size[-1],
                            h=H//self.patch_size[-2],
                            w=W//self.patch_size[-1])

        x = embed = self.patch_embed(x)

        if res != 1:
           
            p0 = self.patch_size[-2]
            p1 = self.patch_size[-1]
            pos_embed = interpolate_pos_encoding(self.pos_embed, T, int(256 // p0 * res), int(256 // p1 * res))
        else:
           
            pos_embed = self._get_pos_embed()

        pos_embed = pos_embed.type_as(x)  # .to(x.device).clone()

        if not self.learn_pos_embed:
            pos_embed = pos_embed.to(x.device).clone().detach()

        x = x + pos_embed
        B, _, C = x.shape

        if not self.interp_noise:
            x_vis = self.interpolate_tensor_with_mask_token(x, mask, mask_token, invert=True)
        else:
            x_vis = x

        if move_patches is not None:

            assert B == 1, "Only support batch size 1 for now"
            for (px, py) in move_patches:
                idx = px * self.w + py
                dx, dy = delta
                nx, ny = px + dx, py + dy
                new_idx = nx * self.w + ny + (self.patch_embed.num_frames - 1) * (self.h * self.w)

                emb = embed[:, idx]
                pos_emb = pos_embed[:, new_idx]
                emb = emb + pos_emb
                x_vis = torch.cat([x_vis, emb[None]], 1)

            if static_patches is not None:
                for (px, py) in static_patches:
                    idx = px * self.w + py
                    new_idx = px * self.w + py + (self.patch_embed.num_frames - 1) * (self.h * self.w)
                    emb = embed[:, idx]
                    pos_emb = pos_embed[:, new_idx]
                    emb = emb + pos_emb
                    x_vis = torch.cat([x_vis, emb[None]], 1)

        for blk_idx, blk in enumerate(self.blocks):
            x_vis = blk(x_vis)
            if blk_idx == return_feat_layer:
                return x_vis

        x_vis = self.norm(x_vis)
        return x_vis

    def _set_inputs(self, *args, **kwargs):
        pass

    def forward(self, x, mask, mask_token, return_feat_layer=None, timestamps=None, move_patches=None, static_patches=None, delta=None, res=1):
        self._set_inputs(x, mask)
        # pass input through the encoder
        x = self.forward_features(x, mask, move_patches, static_patches, delta, mask_token, return_feat_layer=return_feat_layer, res=res)
        # if return_feat_layer is not None and is lesser than the number of blocks it means that we are returning the
        # features of an intermediate block layer. in this case we do not want to apply the head layer
        if return_feat_layer is not None and return_feat_layer < len(self.blocks):
            return x
        # if we are passing through the entire encoder transformer we apply the head layer
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=(16, 16), num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, block_func=Block, block_kwargs={}, k_bias=False
                 ):
        super().__init__()


        self.num_classes = num_classes

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_func(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, **block_kwargs, k_bias=k_bias)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
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
            return self.head(self.norm(x[:,-return_token_num:]))
        elif return_token_num == 0:
            return self.head(self.norm(x))[:,x.size(1):]
        else:
            return self.head(self.norm(x))

    def forward(self, x, return_token_num, return_feat_layer=None):

        # pass input through the decoder        
        for blk_idx, blk in enumerate(self.blocks):
            x = blk(x)
            if blk_idx == return_feat_layer:
                return x

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    default_input_kwargs = {'unnormalize': True}
    def __init__(self,
                 img_size=224, 
                 patch_size=(16, 16),
                 encoder_func=PretrainVisionTransformerEncoder,
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
                 embed_per_frame=False,
                 clumping_factor=None,
                 return_detectron_format=False,
                 interp_noise=False,
                 learn_pos_embed=False,
                 **kwargs
                 ):
        super().__init__()

        self.clumping_factor = clumping_factor

        self.interp_noise = interp_noise

        self.learn_pos_embed = learn_pos_embed

        if self.clumping_factor is not None:
            print('Clumping factor = %d' % self.clumping_factor)
            self.clumping_embed = nn.Conv3d(in_channels=decoder_embed_dim, out_channels=decoder_embed_dim,
                                            kernel_size=(1, clumping_factor, clumping_factor),
                                            stride=(1, clumping_factor, clumping_factor))
            self.clumping_embed.apply(self._init_weights)

            self.up = nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim, kernel_size=2, stride=2)
            self.up.apply(self._init_weights)

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
            embed_per_frame=embed_per_frame,
            block_func=encoder_block_func,
            block_kwargs=encoder_block_kwargs,
            clumping_factor=clumping_factor,
            k_bias=k_bias,
            interp_noise = interp_noise,
            learn_pos_embed=learn_pos_embed,
            **kwargs)

        if not return_detectron_format:
            self.decoder = PretrainVisionTransformerDecoder(
                patch_size=patch_size,
                num_classes= 3*tubelet_size*(patch_size[0]*patch_size[1]) if decoder_num_classes is None else decoder_num_classes,
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

            if not self.interp_noise:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
                trunc_normal_(self.mask_token, std=.02)
            else:
                self.mask_token = None

        self.timestamps = None
        self.encoder.timestamps = None

        if self.learn_pos_embed:
            self.pos_embed = nn.Parameter(get_sinusoid_encoding_table(self.encoder.num_patches, decoder_embed_dim))
        else:
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

        self.return_detectron_format = return_detectron_format

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



    def unpatchify(self, x, mask):
        # Define the input tensor
        B, N, C = x.shape  # batch size
        h, w = self.mask_size[-2:]
        patch_size = self.patch_size[-2:]

        recon = torch.zeros(B, h*w, C).to(x)
        recon[mask[:, -h*w:]] = x.flatten(0, 1)

        rec_imgs = rearrange(recon, 'b n (p c) -> b n p c', c=3)

        rec_imgs = rearrange(rec_imgs,
                             'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
                             p0=1,
                             p1=patch_size[0],
                             p2=patch_size[1],
                             h=h,
                             w=w)

        return rec_imgs


    def forward(self, x, mask, timestamps=None, return_feat_layer=None, res=1, *args, get_encoder_out=False, **kwargs):

        _, _, T, _, _ = x.shape

        self.device = x.device

        enc_out = self.encoder(x, mask, self.mask_token, timestamps=timestamps, return_feat_layer=return_feat_layer, res=res, *args, **kwargs) # [B, N_vis, C_e]

        x_vis = self.encoder_to_decoder(enc_out)

        if return_feat_layer is not None:
            return_feat_layer = return_feat_layer - len(self.encoder.blocks) - 1
            if return_feat_layer < 0:
                return x_vis

        # add pos embedding
        if res != 1:
            p0 = self.patch_size[-2]
            p1 = self.patch_size[-1]
            pos_embed = interpolate_pos_encoding(self.pos_embed, T, int(256 // p0 * res), int(256 // p1 * res))
        else:
            pos_embed = self.pos_embed
        dec_pos_embed = pos_embed.expand(x_vis.size(0), -1, -1).type_as(x)

        if not self.learn_pos_embed:
            dec_pos_embed = dec_pos_embed.to(x.device).clone().detach()

        x_vis = x_vis + dec_pos_embed

        # pass input through the decoder, this will automatically return an intermediate layer if return_feat_layer is set
        x_all = self.decoder(x_vis, 0, return_feat_layer=return_feat_layer)

        if get_encoder_out:
            return x_all, enc_out
        
        return x_all

    def get_intervention_outcome(self, x, move_patches):
        '''
        :param x: input tensor [1, C, T, H, W]: support only batch size 1 for now
        :param move_patches: torch tensor [N, 4] sized array where each row contains patch motion [x1, y1, x2, y2] in pixel coordinates
        :return:
        '''
        B, _, T, H, H = x.shape

        mask = torch.ones(B, self.encoder.hw * self.encoder.num_frames).to(x.device).bool()
        mask[:, :self.encoder.hw * (self.encoder.num_frames - 1)] = False

        move_patches = (move_patches / H) * self.encoder.h
        move_patches = move_patches.to(torch.int64)

        for x1, y1, x2, y2 in move_patches:
            idx2 = x2 * self.encoder.w + y2 + (self.encoder.num_frames - 1) * (self.encoder.h * self.encoder.w)
            mask[:, idx2] = False
            im_x1 = x1 * self.encoder.ph
            im_y1 = y1 * self.encoder.pw
            im_x2 = x2 * self.encoder.ph
            im_y2 = y2 * self.encoder.pw
            x[:, :, -1, im_x2:im_x2 + self.encoder.ph, im_y2:im_y2 + self.encoder.pw] = x[:, :, -2,
                                                                                        im_x1:im_x1 + self.encoder.ph,
                                                                                        im_y1:im_y1 + self.encoder.pw]

        prediction = self.forward(x, mask)[:, -self.encoder.hw:]

        prediction = utils.unpatchify_cwm(
            prediction,
            patch_size=self.encoder.patch_size[-1],
        )  # reshape the output to an image

        return prediction


def pretrain_vit_base_256_scaffold(**kwargs):
    model = PretrainVisionTransformer(
        img_size=256,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_embed_dim=768,
        decoder_num_heads=12,
        decoder_depth=12,
        mlp_ratio=4,
        qkv_bias=True,
        k_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model

