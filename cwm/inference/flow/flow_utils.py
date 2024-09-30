import random
import math
import numpy as np
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn import functional as F
from torchvision import transforms


def compute_optical_flow(embedding_tensor, mask_tensor, frame_size):
    # Unroll the mask tensor and find the indices of the masked and unmasked values in the second frame
    mask_unrolled = mask_tensor.view(-1)

    second_frame_unmask_indices = torch.where(mask_unrolled[frame_size ** 2:] == False)[0]

    # Divide the embedding tensor into two parts: corresponding to the first and the second frame
    first_frame_embeddings = embedding_tensor[0, :frame_size ** 2, :]
    second_frame_embeddings = embedding_tensor[0, frame_size ** 2:, :]

    # Compute the cosine similarity between the unmasked embeddings from the second frame and the embeddings from the first frame
    dot_product = torch.matmul(second_frame_embeddings, first_frame_embeddings.T)
    norms = torch.norm(second_frame_embeddings, dim=1)[:, None] * torch.norm(first_frame_embeddings, dim=1)[None, :]
    cos_sim_matrix = dot_product / norms

    # Find the indices of pixels in the first frame that are most similar to each unmasked pixel in the second frame
    first_frame_most_similar_indices = cos_sim_matrix.argmax(dim=-1)

    # Convert the 1D pixel indices into 2D coordinates
    second_frame_y = second_frame_unmask_indices // frame_size
    second_frame_x = second_frame_unmask_indices % frame_size
    first_frame_y = first_frame_most_similar_indices // frame_size
    first_frame_x = first_frame_most_similar_indices % frame_size

    # Compute the x and y displacements and convert them to float
    displacements_x = (second_frame_x - first_frame_x).float()
    displacements_y = (second_frame_y - first_frame_y).float()

    # Initialize optical flow tensor
    optical_flow = torch.zeros((2, frame_size, frame_size), device=embedding_tensor.device)

    # Assign the computed displacements to the corresponding pixels in the optical flow tensor
    optical_flow[0, second_frame_y, second_frame_x] = displacements_x
    optical_flow[1, second_frame_y, second_frame_x] = displacements_y

    return optical_flow


def get_crops(video_tensor, predictor_image_size):
    B, T, C, H, W = video_tensor.shape

    # Calculate the number of crops needed in both the height and width dimensions
    num_crops_h = math.ceil(H / predictor_image_size) if H > predictor_image_size else 1
    num_crops_w = math.ceil(W / predictor_image_size) if W > predictor_image_size else 1

    # Calculate the step size for the height and width dimensions
    step_size_h = 0 if H <= predictor_image_size else max(0, (H - predictor_image_size) // (num_crops_h - 1))
    step_size_w = 0 if W <= predictor_image_size else max(0, (W - predictor_image_size) // (num_crops_w - 1))

    # Create a list to store the cropped tensors and their start positions
    cropped_tensors = []
    crop_positions = []

    # Iterate over the height and width dimensions, extract the 224x224 crops, and append to the cropped_tensors list
    for i in range(num_crops_h):
        for j in range(num_crops_w):
            start_h = i * step_size_h
            start_w = j * step_size_w
            end_h = min(start_h + predictor_image_size, H)
            end_w = min(start_w + predictor_image_size, W)
            crop = video_tensor[:, :, :, start_h:end_h, start_w:end_w]
            cropped_tensors.append(crop)
            crop_positions.append((start_h, start_w))

    # Reshape the cropped tensors to fit the required input shape of the predictor (B, T, C, predictor_image_size, predictor_image_size)
    cropped_tensors = [crop.reshape(B, T, C, predictor_image_size, predictor_image_size) for crop in cropped_tensors]

    return cropped_tensors, crop_positions


def create_weighted_mask_batched(h, w):
    y_mask = np.linspace(0, 1, h)
    y_mask = np.minimum(y_mask, 1 - y_mask)
    x_mask = np.linspace(0, 1, w)
    x_mask = np.minimum(x_mask, 1 - x_mask)
    weighted_mask = np.outer(y_mask, x_mask)
    return torch.from_numpy(weighted_mask).float()


def join_flow_crops(cropped_tensors, crop_positions, original_shape, predictor_image_size):
    B, T, C, H, W = original_shape

    # Initialize an empty tensor to store the reconstructed video
    reconstructed_video = torch.zeros((B, T, C, H, W)).to(cropped_tensors[0].device)

    # Create a tensor to store the sum of weighted masks
    weighted_masks_sum = torch.zeros((B, T, C, H, W)).to(cropped_tensors[0].device)

    # Create a weighted mask for the crops
    weighted_mask = create_weighted_mask_batched(predictor_image_size, predictor_image_size).to(cropped_tensors[0].device)
    weighted_mask = weighted_mask[None, None, None, :, :]  # Extend dimensions to match the cropped tensor.

    for idx, crop in enumerate(cropped_tensors):
        start_h, start_w = crop_positions[idx]

        # Multiply the crop with the weighted mask
        weighted_crop = crop * weighted_mask

        # Add the weighted crop to the corresponding location in the reconstructed_video tensor
        reconstructed_video[:, :, :, start_h:(start_h + predictor_image_size), start_w:(start_w + predictor_image_size)] += weighted_crop

        # Update the weighted_masks_sum tensor
        weighted_masks_sum[:, :, :, start_h:(start_h + predictor_image_size), start_w:(start_w + predictor_image_size)] += weighted_mask

    # Add a small epsilon value to avoid division by zero
    epsilon = 1e-8

    # Normalize the reconstructed video by dividing each pixel by its corresponding weighted_masks_sum value plus epsilon
    reconstructed_video /= (weighted_masks_sum + epsilon)

    return reconstructed_video


def l2_norm(x):
    return x.square().sum(-3, True).sqrt()


resize = lambda x, a: F.interpolate(x, [int(a * x.shape[-2]), int(a * x.shape[-1])], mode='bilinear', align_corners=False)

upsample = lambda x, H, W: F.interpolate(x, [int(H), int(W)], mode='bilinear', align_corners=False)


def get_occ_masks(flow_fwd, flow_bck, occ_thresh=0.5):
    fwd_bck_cycle, _ = backward_warp(img2=flow_bck, flow=flow_fwd)
    flow_diff_fwd = flow_fwd + fwd_bck_cycle

    bck_fwd_cycle, _ = backward_warp(img2=flow_fwd, flow=flow_bck)
    flow_diff_bck = flow_bck + bck_fwd_cycle

    norm_fwd = l2_norm(flow_fwd) ** 2 + l2_norm(fwd_bck_cycle) ** 2
    norm_bck = l2_norm(flow_bck) ** 2 + l2_norm(bck_fwd_cycle) ** 2

    occ_thresh_fwd = occ_thresh * norm_fwd + 0.5
    occ_thresh_bck = occ_thresh * norm_bck + 0.5

    occ_mask_fwd = 1 - (l2_norm(flow_diff_fwd) ** 2 > occ_thresh_fwd).float()
    occ_mask_bck = 1 - (l2_norm(flow_diff_bck) ** 2 > occ_thresh_bck).float()

    return occ_mask_fwd, occ_mask_bck


def forward_backward_cycle_consistency(flow_fwd, flow_bck, niters=10):
    # Make sure to be using axes-swapped, upsampled flows!
    bck_flow_clone = flow_bck.clone().detach()
    fwd_flow_clone = flow_fwd.clone().detach()

    for i in range(niters):
        fwd_bck_cycle_orig, _ = backward_warp(img2=bck_flow_clone, flow=fwd_flow_clone)
        flow_diff_fwd_orig = fwd_flow_clone + fwd_bck_cycle_orig

        fwd_flow_clone = fwd_flow_clone - flow_diff_fwd_orig / 2

        bck_fwd_cycle_orig, _ = backward_warp(img2=fwd_flow_clone, flow=bck_flow_clone)
        flow_diff_bck_orig = bck_flow_clone + bck_fwd_cycle_orig

        bck_flow_clone = bck_flow_clone - flow_diff_bck_orig / 2

    return fwd_flow_clone, bck_flow_clone


from PIL import Image


def resize_flow_map(flow_map, target_size):
    """
    Resize a flow map to a target size while adjusting the flow vectors.

    Parameters:
    flow_map (numpy.ndarray): Input flow map of shape (H, W, 2) where each pixel contains a (dx, dy) flow vector.
    target_size (tuple): Target size (height, width) for the resized flow map.

    Returns:
    numpy.ndarray: Resized and scaled flow map of shape (target_size[0], target_size[1], 2).
    """
    # Get the original size
    flow_map = flow_map[0].detach().cpu().numpy()
    flow_map = flow_map.transpose(1, 2, 0)
    original_size = flow_map.shape[:2]

    # Separate the flow map into two channels: dx and dy
    flow_map_x = flow_map[:, :, 0]
    flow_map_y = flow_map[:, :, 1]

    # Convert each flow channel to a PIL image for resizing
    flow_map_x_img = Image.fromarray(flow_map_x)
    flow_map_y_img = Image.fromarray(flow_map_y)

    # Resize both channels to the target size using bilinear interpolation
    flow_map_x_resized = flow_map_x_img.resize(target_size, Image.BILINEAR)
    flow_map_y_resized = flow_map_y_img.resize(target_size, Image.BILINEAR)

    # Convert resized PIL images back to NumPy arrays
    flow_map_x_resized = np.array(flow_map_x_resized)
    flow_map_y_resized = np.array(flow_map_y_resized)

    # Compute the scaling factor based on the size change
    scale_factor = target_size[0] / original_size[0]  # Scaling factor for both dx and dy

    # Scale the flow vectors (dx and dy) accordingly
    flow_map_x_resized *= scale_factor
    flow_map_y_resized *= scale_factor

    # Recombine the two channels into a resized flow map
    flow_map_resized = np.stack([flow_map_x_resized, flow_map_y_resized], axis=-1)

    flow_map_resized = torch.from_numpy(flow_map_resized)[None].permute(0, 3, 1, 2)

    return flow_map_resized


def mean_filter_flow(tensor):
    C, H, W = tensor.shape

    # Create zero-filled tensors for the shifted crops
    down_shifted = torch.zeros_like(tensor)
    up_shifted = torch.zeros_like(tensor)
    right_shifted = torch.zeros_like(tensor)
    left_shifted = torch.zeros_like(tensor)

    # Shift the tensor and store the results in the zero-filled tensors
    down_shifted[:, :H - 1, :] = tensor[:, 1:, :]
    up_shifted[:, 1:, :] = tensor[:, :H - 1, :]
    right_shifted[:, :, :W - 1] = tensor[:, :, 1:]
    left_shifted[:, :, 1:] = tensor[:, :, :W - 1]

    # Average the tensor with its four crops
    result = (tensor + down_shifted + up_shifted + right_shifted + left_shifted) / 5.0

    return result


def extract_optical_flow(predictor,
                         mask_generator,
                         img1,
                         img2,
                         conditioning_img=None,
                         num_scales=1,
                         min_scale=400,
                         num_mask_samples=100,
                         mean_filter=True):

    '''
    Extract optical flow between two images using the given predictor.
    :param predictor: CWM model
    :param mask_generator: PytorchMaskGeneratorWrapper object for generating masks
    :param img1: input image 1 [B, C, H, W]
    :param img2: input image 2 [B, C, H, W]
    :param conditioning_img: image used for conditioning (in case predictor is a 3 frame model) input image 3 [B, C, H, W]
    :param num_scales: number of scales at which to compute flow
    :param min_scale: minimum scale at which to compute flow. Flow will be computed for a range of scales (specified by the num_scales argument) from min_scale to img1.shape[-2]
    :param num_mask_samples: number of random samples of the target frame mask to use for getting a better statistical estimate of the optical flow
    :param mean_filter: use a mean filter to improve quality of flows
    :return: optical flow of shape [B, 2, H, W]
    '''

    B = img1.shape[0]
    h1 = img2.shape[-2]
    w1 = img2.shape[-1]

    alpha = (min_scale / img1.shape[-2]) ** (1 / (num_scales - 1)) if num_scales > 1 else 1

    predictor_image_size = predictor.image_size[0]

    frame_size = predictor_image_size // predictor.patch_size[-1]

    num_frames = predictor.num_frames

    #save the optical flows at each scale
    all_flows = []

    #save the scale factors
    scales_h = []
    scales_w = []

    #Compute flow at multiple scales and average them. As the predictor only takes in images of size predictor_image_size, we compute the optical flow by dividing the input images into crops of size predictor_image_size and joining them
    for aidx in range(num_scales):

        img1_scaled = F.interpolate(img1.clone(), [int((alpha ** aidx) * h1), int((alpha ** aidx) * w1)], mode='bicubic', align_corners=True)
        img2_scaled = F.interpolate(img2.clone(), [int((alpha ** aidx) * h1), int((alpha ** aidx) * w1)], mode='bicubic', align_corners=True)

        # If conditioning image is provided (in case of a three frame model), scale it to the same size as img1 and img2
        if conditioning_img is not None:
            conditioning_img_scaled = F.interpolate(conditioning_img.clone(),
                                                    [int((alpha ** aidx) * h1), int((alpha ** aidx) * w1)],
                                                    mode='bilinear', align_corners=False)
        
        # save the scale factors for each scale: to be used later for resize the flow map
        h2 = img2_scaled.shape[-2]
        w2 = img2_scaled.shape[-1]

        scales_h.append(h1 / h2)
        scales_w.append(w1 / w2)
        
        # concatenate the images before extracting crops
        if conditioning_img is not None:
            video = torch.cat(
                [conditioning_img_scaled.unsqueeze(1), img2_scaled.unsqueeze(1), img1_scaled.unsqueeze(1)], 1)
        else:
            #stack images in reverse order -- we will compute backward flow and then reverse it get forward flow
            video = torch.cat([img2_scaled.unsqueeze(1)] * (num_frames - 1) + [img1_scaled.unsqueeze(1)], 1)


        crops, crop_positions = get_crops(video, predictor_image_size)
        num_crops = len(crops)

        #process all the crops in batched mode
        crop = torch.cat(crops, 0).to(video.device)

        #for storing the optical flows as they are computed
        optical_flows = torch.zeros(B * num_crops, 2, frame_size, frame_size).to(video.device)

        #tracking the patches for which flow has been computed
        mask_counts = torch.zeros(frame_size, frame_size).to(video.device)

        i = 0
        #repeat until flow for all patches is computed or number of sampling steps is reached
        while i < num_mask_samples or (mask_counts == 0).any().item():

            mask = mask_generator().bool().to(video.device)

            # track the number of patches that were unmasked in the last frame
            last_frame_mask = ~mask[0, (frame_size * frame_size) * (num_frames - 1):]
            mask_counts += last_frame_mask.reshape(frame_size, frame_size)

            with torch.cuda.amp.autocast(enabled=True):

                processed_x = crop.transpose(1, 2)

                #get features
                encoder_out = predictor.encoder(processed_x.to(torch.float16), mask.repeat(B * num_crops, 1))
                encoder_to_decoder = predictor.encoder_to_decoder(encoder_out)
                encoder_to_decoder = encoder_to_decoder[:, (frame_size * frame_size) * (num_frames - 2):, :]

                # keep track of the flow_mask: flows are valid only for those patches that are visible in the target frame
                flow_mask = mask[:, (frame_size * frame_size) * (num_frames - 2):]

                optical_flow = []

                for b in range(B * num_crops):
                    #compute optical flow
                    batch_flow = compute_optical_flow(encoder_to_decoder[b].unsqueeze(0), flow_mask, frame_size)

                    #apply a low pass filter to remove artifacts
                    if mean_filter:
                        batch_flow = mean_filter_flow(batch_flow)

                    optical_flow.append(batch_flow.unsqueeze(0))

                optical_flow = torch.cat(optical_flow, 0)

                #record the optical flows
                optical_flows += optical_flow

            i += 1

        #average the optical flows computed for various random samples of the target frame mask
        optical_flows = optical_flows / mask_counts

        crop_flows = optical_flows.split(B, 0)

        #resize crops to predictor_image_size dimensions
        resized_crops = [F.interpolate(flow, [predictor_image_size, predictor_image_size], mode='bicubic', align_corners=True).unsqueeze(1).cpu() for flow in crop_flows]

        #join flow crops to get the flow of the full image
        optical_flows_joined = join_flow_crops(resized_crops, crop_positions, (B, 1, 2, video.shape[-2], video.shape[-1]), predictor_image_size).squeeze(1)

        all_flows.append(optical_flows_joined)

    all_flows_rescaled = []

    for scale_idx, flow in enumerate(all_flows):

        orig_img_to_scaled_img_factor_h = scales_h[scale_idx]
        orig_img_to_scaled_img_factor_w = scales_w[scale_idx]
        patch_size = predictor.patch_size[-1]

        #make flow at all scales are the same size as the flow at the original input scale
        flow = F.interpolate(flow, [int(all_flows[0].shape[-2]), int(all_flows[0].shape[-1])], mode='bicubic', align_corners=True)

        scaled_new_r = torch.zeros_like(flow)

        #scale the flow values back to the original image resolution. First scale from patch resolution to predictor_image_size and then from predictor_image_size to original image
        scaled_new_r[:, 0, :, :] = flow[:, 0, :, :] * patch_size
        scaled_new_r[:, 0, :, :] = scaled_new_r[:, 0, :, :] * orig_img_to_scaled_img_factor_h

        scaled_new_r[:, 1, :, :] = flow[:, 1, :, :] * patch_size
        scaled_new_r[:, 1, :, :] = scaled_new_r[:, 1, :, :] * orig_img_to_scaled_img_factor_w

        all_flows_rescaled.append(scaled_new_r.unsqueeze(-1))

    optical_flow = torch.cat(all_flows_rescaled, -1).mean(-1)

    #reverse the backward flow to get forward flow
    optical_flow = -optical_flow

    return optical_flow


def extract_jacobians_and_flows(img1, img2,
                                flow_generator,
                                mask,
                                target_mask=None):
    IMAGE_SIZE = img1.shape[-2:]

    y = torch.cat([img2.unsqueeze(1), img1.unsqueeze(1)], 1)

    jacobians, flows, _ = flow_generator(y, mask, target_mask)

    # swap x,y flow dims
    flows = torch.cat([flows[0, 1].unsqueeze(0), flows[0, 0].unsqueeze(0)])

    # upsample to 224
    flows = flows.unsqueeze(0).repeat_interleave(IMAGE_SIZE[0] // flows.shape[-1], -1).repeat_interleave(
        IMAGE_SIZE[0] // flows.shape[-1], -2)

    return jacobians, flows


def boltzmann(x, beta=1, eps=1e-9):
    if beta is None:
        return x
    x = torch.exp(x * beta)
    return x / x.amax((-1, -2), keepdim=True).clamp(min=eps)


def imagenet_normalize(x, temporal_dim=1):
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(x.device)[None, None, :, None, None].to(x)
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(x.device)[None, None, :, None, None].to(x)
    if temporal_dim == 2:
        mean = mean.transpose(1, 2)
        std = std.transpose(1, 2)
    return (x - mean) / std


def imagenet_unnormalize(x, temporal_dim=2):
    device = x.device
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, None, :, None, None].to(x)
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, None, :, None, None].to(x)
    if temporal_dim == 2:
        mean = mean.transpose(1, 2)
        std = std.transpose(1, 2)
    x = x * std + mean
    return x


def coordinate_ims(batch_size, seq_length, imsize, normalize=True, dtype_out=torch.float32):
    static = False
    if seq_length == 0:
        static = True
        seq_length = 1
    B = batch_size
    T = seq_length
    H, W = imsize
    ones = torch.ones([B, H, W, 1], dtype=dtype_out)
    if normalize:
        h = torch.divide(torch.arange(H).to(ones), torch.tensor(H - 1, dtype=dtype_out))
        h = 2.0 * ((h.view(1, H, 1, 1) * ones) - 0.5)
        w = torch.divide(torch.arange(W).to(ones), torch.tensor(W - 1, dtype=dtype_out))
        w = 2.0 * ((w.view(1, 1, W, 1) * ones) - 0.5)
    else:
        h = torch.arange(H).to(ones).view(1, H, 1, 1) * ones
        w = torch.arange(W).to(ones).view(1, 1, W, 1) * ones
    h = torch.stack([h] * T, 1)
    w = torch.stack([w] * T, 1)
    hw_ims = torch.cat([h, w], -1)
    if static:
        hw_ims = hw_ims[:, 0]
    return hw_ims


def get_distribution_centroid(dist, eps=1e-9, normalize=False):
    B, T, C, H, W = dist.shape
    assert C == 1
    dist_sum = dist.sum((-2, -1), keepdim=True).clamp(min=eps)
    dist = dist / dist_sum

    grid = coordinate_ims(B, T, [H, W], normalize=normalize).to(dist.device)
    grid = grid.permute(0, 1, 4, 2, 3)
    centroid = (grid * dist).sum((-2, -1))
    return centroid


def sampling_grid(height, width):
    H, W = height, width
    grid = torch.stack([
        torch.arange(W).view(1, -1).repeat(H, 1),
        torch.arange(H).view(-1, 1).repeat(1, W)
    ], -1)
    grid = grid.view(1, H, W, 2)
    return grid


def normalize_sampling_grid(coords):
    assert len(coords.shape) == 4, coords.shape
    assert coords.size(-1) == 2, coords.shape
    H, W = coords.shape[-3:-1]
    xs, ys = coords.split([1, 1], -1)
    xs = 2 * xs / (W - 1) - 1
    ys = 2 * ys / (H - 1) - 1
    return torch.cat([xs, ys], -1)


def backward_warp(img2, flow, do_mask=False):
    """
    Grid sample from img2 using the flow from img1->img2 to get a prediction of img1.

    flow: [B,2,H',W'] in units of pixels at its current resolution. The two channels
          should be (x,y) where larger y values correspond to lower parts of the image.
    """

    ## resize the flow to the image size.
    ## since flow has units of pixels, its values need to be rescaled accordingly.
    if list(img2.shape[-2:]) != list(flow.shape[-2:]):
        scale = [img2.size(-1) / flow.size(-1),  # x
                 img2.size(-2) / flow.size(-2)]  # y
        scale = torch.tensor(scale).view(1, 2, 1, 1).to(flow.device)
        flow = scale * transforms.Resize(img2.shape[-2:])(flow)  # defaults to bilinear

    B, C, H, W = img2.shape

    ## use flow to warp sampling grid
    grid = sampling_grid(H, W).to(flow.device) + flow.permute(0, 2, 3, 1)

    ## put grid in normalized image coordinates
    grid = normalize_sampling_grid(grid)

    ## backward warp, i.e. sample pixel (x,y) from (x+flow_x, y+flow_y)
    img1_pred = F.grid_sample(img2, grid, align_corners=True)

    if do_mask:
        mask = (grid[..., 0] > -1) & (grid[..., 0] < 1) & \
               (grid[..., 1] > -1) & (grid[..., 1] < 1)
        mask = mask[:, None].to(img2.dtype)
        return (img1_pred, mask)

    else:
        return (img1_pred, torch.ones_like(grid[..., 0][:, None]).float())
