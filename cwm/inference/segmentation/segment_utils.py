import torch
from external.raft_interface import RAFTInterface, FlowSampleFilter
import cwm.utils as utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
flow_interface = RAFTInterface()
flow_sample_filter = FlowSampleFilter()
DEBUG = False
np.random.seed(42)
torch.manual_seed(42)


def get_segment_at_pos(model, img, pos, num_samples=16, num_static=2, max_movement=4, threshold=0.2):

    '''
    Extract single segment from a static input image based on the sampling distribution
    '''

    ## Step 0: Preprocessing
    # hence, we create a static video by replicating the input image in temporal dimension
    x = img.unsqueeze(2).expand(-1, -1, model.num_frames, -1, -1)
    B, C, T, H, W = x.shape
    model.adjust_input_resolution(H, W)

    ## Step 1: define the distribution for sampling static locations (using predominance)
    sampling_dist = get_motion_sampling_distribution(
        x, img_size=model.image_size, dist_size=model.encoder.dims
    )

    # sample initial moving and static locations from the distribution
    flow_mag = get_motion_samples_at_pos(
        model, x, pos, sampling_dist, num_samples=num_samples, num_static=num_static, max_movement=max_movement
    )

    segment = utils.normalize_flow_mag(flow_mag.mean(1)) >= threshold

    return segment

def get_single_segment(model, img, sampling_dist=None, num_iters=4, num_samples=6, max_movement=4):

    '''
    Extract single segment from a static input image based on the sampling distribution
    '''

    ## Step 0: Preprocessing
    # hence, we create a static video by replicating the input image in temporal dimension
    x = img.unsqueeze(2).expand(-1, -1, model.num_frames, -1, -1)
    B, C, T, H, W = x.shape
    model.adjust_input_resolution(H, W)

    ## Step 1: define the distribution for sampling moving and static locations
    sampling_dist = get_motion_sampling_distribution(
        x, img_size=model.image_size, dist_size=model.encoder.dims
    ) if sampling_dist is None else sampling_dist

    ## Step 2: sample initial moving and static locations from the distribution
    init_move_pos, init_static_pos, init_flow_mag = get_motion_samples(
        model, x, sampling_dist, num_samples=num_samples, max_movement=max_movement
    )

    ## Step 3: iteratively add more moving and static locations to refine the segment
    flow_mag = refine_motion_samples(
        model, x, init_move_pos, init_static_pos, init_flow_mag,
        num_samples=num_samples, num_iters=num_iters, max_movement=max_movement
    )
    segment = (utils.normalize_flow_mag(flow_mag) > 0.1).float().mean(1) > 0.5

    return [segment[0].cpu()]



def get_motion_sampling_distribution(x, img_size, dist_size):
    '''
    Compute a distribution for sampling patches to move and generate counterfactual model
    '''

    sampling_dist = utils.get_dino_predominance(x[:, :, 0], dims=[60, 60], img_size=img_size)[0]
    sampling_dist = F.interpolate(sampling_dist, dist_size, mode='bilinear', align_corners=False)
    sampling_dist = sampling_dist.squeeze(1)  # ** 4
    return sampling_dist


def get_motion_samples(model, x, sampling_dist, num_samples, max_movement):
    B, C, T, H, W = x.shape
    N = num_samples
    dims = torch.tensor([model.encoder.h, model.encoder.w]).view(1, 1, 2).to(x.device)

    num_valid_init_samples, sampling_count = 0, 0
    init_move_pos, init_static_pos, init_flow_mag, max_score = None, None, None, 0

    while num_valid_init_samples < N and sampling_count < 32:

        if DEBUG:
            print('Getting motion samples', num_valid_init_samples, sampling_count)

        # increment the sampling count to avoid infinite loop
        sampling_count += 1

        # sample one moving position per example in the batch
        move_pos = utils.sample_positions_from_dist(size=[1, 1], dist=sampling_dist).repeat(N, 1, 1)  # [BN, 1, 2]

        # each move position has N static positions and movement directions
        static_pos = utils.sample_positions_from_dist(size=[B * N, 1], dist=-sampling_dist)  # [BN, 1, 2]

        ## compute initial flow maps of counterfactual motions
        _x = x.repeat(N, 1, 1, 1, 1)  # [BN, C, T, H, W]
        pred = model.get_directional_intervention_outcome(_x, move_pos=move_pos, static_pos=static_pos, max_movement=max_movement)
        flow = flow_interface(_x[:, :, 0].float(), pred.clamp(0, 1).float()).view(B, N, 2, H, W)  # [B, N, 2, H, W]

        # filter out invalid flow samples
        # normalized_move_pos = move_pos.view(B, N, 2) / dims * 2 - 1 # normalize to range [-1, 1]
        # flow, _ = flow_sample_filter(flow, normalized_move_pos) # [B, N, 2, H, W]

        flow_mag = flow.norm(p=2, dim=2).view(B, N, H, W)

        # filter out invalid samples of motion position
        # if over half of the flow samples result in trivial motion, discard the current motion position
        avg_flow_mag = flow_mag.flatten(2, 3).mean(-1)  # [B, N]
        valid = avg_flow_mag > 0.1  # [B, N] valid if there is non-trivial motion
        if valid.float().mean() < 0.5:
            continue

        valid_flow_mag = flow_mag[valid].unsqueeze(0)
        scores = utils.compute_pairwise_flow_mag_overlap(valid_flow_mag, valid_flow_mag)
        num_valid_init_samples += 1

        if DEBUG:
            print('scores of sample', scores)

        if scores > max_score:
            init_move_pos, init_static_pos, init_flow_mag, max_score = move_pos, static_pos, flow_mag, scores

        if DEBUG: # visualize initial samples (for debugging purpose)
            print(flow.shape, move_pos.shape, static_pos.shape)
            vis_motion_samples(flow[0], move_pos, static_pos, patch_size=model.encoder.patch_size[-1])

    return init_move_pos, init_static_pos, init_flow_mag

def get_motion_samples_at_pos(model, x, pos, sampling_dist, num_samples, max_movement, num_static=2):
    B, C, T, H, W = x.shape
    N = num_samples
    dims = torch.tensor([model.encoder.h, model.encoder.w]).view(1, 1, 2).to(x.device)
    img_dims = torch.tensor([H, W]).view(1, 1, 2).to(x.device)

    # normalize query postions to be moved
    move_pos = torch.tensor(pos).to(x.device).view(1, -1, 2).repeat(N, 1, 1)  # [BN, 1, 2]
    move_pos = move_pos / img_dims * dims

    # each move position has N static positions and movement directions
    static_pos = utils.sample_positions_from_dist(size=[B * N, num_static], dist=-sampling_dist)  # [BN, 1, 2]

    # compute initial flow maps of counterfactual motions
    _x = x.repeat(N, 1, 1, 1, 1)  # [BN, C, T, H, W]
    pred = model.get_directional_intervention_outcome(_x, move_pos=move_pos, static_pos=static_pos, max_movement=max_movement)
    flow = flow_interface(_x[:, :, 0].float(), pred.clamp(0, 1).float()).view(B, N, 2, H, W)  # [B, N, 2, H, W]

    # filter out invalid flow samples
    normalized_move_pos = move_pos.view(B, N, 2) / dims * 2 - 1 # normalize to range [-1, 1]
    flow, _ = flow_sample_filter(flow, normalized_move_pos) # [B, N, 2, H, W]
    flow_mag = flow.norm(p=2, dim=2).view(B, N, H, W)

    # filter out invalid samples of motion position
    # if over half of the flow samples result in trivial motion, discard the current motion position
    avg_flow_mag = flow_mag.flatten(2, 3).mean(-1)  # [B, N]
    valid = avg_flow_mag > 0.1  # [B, N] valid if there is non-trivial motion
    valid_flow_mag = flow_mag[valid].unsqueeze(0)

    if DEBUG: # visualize initial samples (for debugging purpose)
        print(flow.shape, move_pos.shape, static_pos.shape)
        vis_motion_samples(flow[0], move_pos, static_pos, patch_size=model.encoder.patch_size[-1])

    return valid_flow_mag

def refine_motion_samples(
        model,
        x,
        init_move_pos,
        init_static_pos,
        init_flow_mag,
        num_samples,
        num_iters,
        max_movement,
        merge_thresh=0.6
):
    B, C, T, H, W = x.shape
    N = num_samples
    _x = x.repeat(N, 1, 1, 1, 1)  # [BN, C, T, H, W]
    prev_flow_mag = init_flow_mag

    npos_per_iter = 1
    prev_move_pos = init_move_pos.repeat(1, npos_per_iter, 1)
    prev_static_pos = init_static_pos.repeat(1, npos_per_iter, 1)
    for it in range(num_iters):
        sampling_dist = F.interpolate(prev_flow_mag, size=model.encoder.dims, mode='bilinear').mean(1)
        # sample one moving position per example in the batch
        move_pos = utils.sample_positions_from_dist(size=[1, npos_per_iter], dist=sampling_dist).repeat(N, 1, 1)
        move_pos = torch.cat([prev_move_pos, move_pos], dim=1)

        # each move position has N static positions and movement directions
        static_pos = utils.sample_positions_from_dist(size=[B * N, npos_per_iter], dist=-sampling_dist)  # [BN, 1, 2]
        static_pos = torch.cat([prev_static_pos, static_pos], dim=1)

        pred = model.get_directional_intervention_outcome(_x, move_pos=move_pos, static_pos=static_pos, max_movement=max_movement)
        flow, flow_mag = flow_interface(_x[:, :, 0].float(), pred.clamp(0, 1).float(), return_magnitude=True)
        overlap = utils.compute_flow_mag_overlap(flow_mag, prev_flow_mag[0])  # [N,]

        merge = (overlap > merge_thresh)[:, None, None]  # [N, 1, 1]
        if DEBUG:
            print('Iteration', it, 'merge', merge, overlap)

        prev_move_pos = torch.where(merge, move_pos,
                                    torch.cat([prev_move_pos, prev_move_pos[:, -npos_per_iter:]], dim=1))
        prev_static_pos = torch.where(merge, static_pos,
                                      torch.cat([prev_static_pos, prev_static_pos[:, -npos_per_iter:]], dim=1))

        prev_flow_mag = flow_mag.unsqueeze(0)

        # visualize samples
        if DEBUG:
            vis_motion_samples(flow, prev_move_pos, prev_static_pos, patch_size=model.encoder.patch_size[-1])
            # patch_size = model.encoder.patch_size[-1]
            # fig, axs = plt.subplots(1, num_samples, figsize=(2 * num_samples, 2 * num_samples))
            #
            # for i in range(num_samples):
            #     flow_rgb = utils.flow_to_rgb(flow[i].cpu().permute(1, 2, 0))
            #
            #     axs[i].imshow(flow_rgb)
            #     axs[i].set_axis_off()
            #     for k in range(move_pos.shape[1]):
            #         move = prev_move_pos[i, k].cpu()
            #         static = prev_static_pos[i, k].cpu()
            #         axs[i].scatter(move[1] * patch_size, move[0] * patch_size, color='green', s=20)
            #         axs[i].scatter(static[1] * patch_size, static[0] * patch_size, color='red', s=20)
            # fig.subplots_adjust(wspace=0.01, hspace=0.01)  # Change these values to adjust space
            #
            # plt.show()
            # plt.close()

    return prev_flow_mag


def vis_motion_samples(flow, move_pos=None, static_pos=None, patch_size=None):
    num_samples = flow.shape[0]
    fig, axs = plt.subplots(1, num_samples, figsize=(2 * num_samples, 2 * num_samples))

    for i in range(num_samples):
        flow_rgb = utils.flow_to_rgb(flow[i].cpu().permute(1, 2, 0))
        axs[i].imshow(flow_rgb)
        if move_pos is not None and static_pos is not None:
            for k in range(move_pos.shape[1]):
                move = move_pos[i, k].cpu()
                axs[i].scatter(move[1] * patch_size, move[0] * patch_size, color='green', s=20)
            for k in range(static_pos.shape[1]):
                static = static_pos[i, k].cpu()
                axs[i].scatter(static[1] * patch_size, static[0] * patch_size, color='red', s=20)
        axs[i].set_axis_off()

    fig.subplots_adjust(wspace=0.01, hspace=0.01)  # Change these values to adjust space
    plt.show()
    plt.close()
