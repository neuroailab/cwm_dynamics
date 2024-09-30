from einops import rearrange
import torch
import numpy as np
from torchvision import transforms

def unpatchify(labels, norm=True):
    # Define the input tensor
    B = labels.shape[0]  # batch size
    N_patches = int(np.sqrt(labels.shape[1]))  # number of patches along each dimension
    patch_size = int(np.sqrt(labels.shape[2] / 3))  # patch size along each dimension
    channels = 3  # number of channels

    rec_imgs = rearrange(labels, 'b n (p c) -> b n p c', c=3)
    #To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
    rec_imgs = rearrange(rec_imgs,
                         'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
                         p0=1,
                         p1=patch_size,
                         p2=patch_size,
                         h=N_patches,
                         w=N_patches)
    if norm:
        MEAN = torch.from_numpy(np.array((0.485, 0.456, 0.406))[None, :, None, None, None]).to(labels.device).half()
        STD = torch.from_numpy(np.array((0.229, 0.224, 0.225))[None, :, None, None, None]).to(labels.device).half()

        rec_imgs = (rec_imgs - MEAN) / STD

    return rec_imgs

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

def get_keypoints_batch(model, x,
                        n_samples,
                        n_rounds,
                        mask=None,
                        pool='avg',
                        ):
    '''
    :param model: and instance of the CWM model class
    :param x: input pair or triplet of frames of size [B, T, C, H, W]
    :param n_samples: number of potential candidates to look at on each round
                      (produces one new unmasked per round)
    :param n_rounds: number of keypoints to extract
    :param mask: an initial mask indicating any keypoints already found
    :param pool: which aggregation function to use for pooling the raw error map to get a per-patch error estimate
    :return:
           mask: final mask with discovered keypoint patch locations set to True [B, number_of_patches]
           choices: x, y indices of keypoints [B, n_rounds, 2]
           err_array: the error maps for every keypoint found List [[B, n_samples, H, W]]*n_rounds
           feat: the features at the final state of the mask [B, C, H, W]
           keypoint_recon: the final reconstruction of the target frame (of the first element in the batch) given the keypoint patches [H, W, C]
    '''

    B = x.shape[0]

    IMAGE_SIZE = [224, 224]
    predictor = model
    patch_size = predictor.patch_size[-1]
    num_frames = predictor.num_frames
    patch_num = IMAGE_SIZE[0] // patch_size
    # this is setup for getting per-patch error
    if pool == 'avg':
        pool_op = torch.nn.AvgPool2d(patch_size, stride=patch_size)
    elif pool == 'max':
        pool_op = torch.nn.MaxPool2d(patch_size, stride=patch_size)

    n_patches = patch_num * patch_num

    # initializing mask at the fully masked state
    mshape = num_frames * patch_num * patch_num
    mshape_masked = (num_frames - 1) * patch_num * patch_num

    if mask is None:
        mask = torch.ones([B, mshape], dtype=torch.bool)
        mask[:, :mshape_masked] = False

    err_array = []
    choices = []

    for round_num in range(n_rounds):
        # get the current prediction with current state of the mask
        out = unpatchify(predictor(x, mask, forward_full=True))

        keypoint_recon = out.clone()

        # get the error map
        err_mat = (out[:, :, 0] - x[:, :, -1]).abs().mean(1)
        # pool it to patch-size
        pooled_err = pool_op(err_mat[:, None])
        # flatten the rror
        flat_pooled_error = pooled_err.flatten(1, 3)
        # set error to be zero where the mask is unmasked so it doesn't interfere
        flat_pooled_error[mask[:, -n_patches:] == False] = 0
        # sort patches by where the error is highest
        err_sort = torch.argsort(flat_pooled_error, -1)
        new_mask = mask.clone().detach()
        errors = []
        tries = []
        err_choices = 0

        # look at various candidates to reveal in the next round
        for sample_num in range(n_samples):

            err_choices += 1
            new_try = (num_frames - 1)  * n_patches + err_sort[:, -1 * err_choices]
            tries.append(new_try)

            for k in range(B):
                new_mask[k, new_try[k]] = False

            reshaped_new_mask = upsample_masks(
                new_mask.view(B, num_frames, IMAGE_SIZE[1] // patch_size, IMAGE_SIZE[1] // patch_size)[:, (num_frames - 1):],
                IMAGE_SIZE)[:, 0]

            out = unpatchify(predictor(x, new_mask, forward_full=True))

            abs_error = (out[:, :, 0] - x[:, :, -1]).abs().sum(1).cpu()

            masked_abs_error = abs_error * reshaped_new_mask
            error = masked_abs_error.flatten(1, 2).sum(-1)
            errors.append(error)

            # take the best one
            for k in range(B):
                new_mask[k, new_try[k]] = True

        errors = torch.stack(errors, 1)
        tries = torch.stack(tries, 1)
        best_ind = torch.argmin(errors, dim=-1)
        best = torch.tensor([tries[k, best_ind[k]] for k in range(B)])
        choices.append(best)
        err_array.append(errors)

        for k in range(B):
            mask[k, best[k]] = False

    feat = predictor(x, mask, forward_full=True, return_features=True)

    feat = feat

    choices = torch.stack(choices, 1)

    choices = choices % mshape_masked
    choices_x = choices % (patch_num)
    choices_y = choices // (patch_num)
    choices = torch.stack([choices_x, choices_y], 2)

    out = unpatchify(predictor(x, mask, forward_full=True), norm=False)

    keypoint_recon = out[0, :, 0].permute(1, 2, 0).detach().cpu().numpy() * 255

    return mask, choices, err_array, feat, keypoint_recon.astype('uint8')