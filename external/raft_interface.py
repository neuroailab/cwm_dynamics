import sys
import os
sys.path.insert(0, os.path.join(os.environ['HOME'], '.cache/torch', 'RAFT/core'))
from raft import RAFT
from utils import flow_viz
sys.path = sys.path[1:] # remove the first path to RAFT
import torch
from cwm.utils import imagenet_unnormalize, sample_embedding
from torch import nn
import argparse


class Args:
    model = os.path.join(os.environ['HOME'], '.cache/torch', 'RAFT/models/raft-sintel.pth')
    small = False
    path = None
    mixed_precision = False
    alternate_corr = False

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

class RAFTInterface(nn.Module):
    def __init__(self):
        super().__init__()
        args = Args()
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        self.model = model.module
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def prepare_inputs(x):
        # make sure the input is in the correct format for RAFT
        if x.max() <= 1.0 and x.min() >= 0.: # range(0, 1)
            x = x * 255.
        elif x.min() < 0: # imagenet normalized:
            x = imagenet_unnormalize(x)
            x = x * 255.

        return x

    def forward(self, x0, x1, return_magnitude=False):
        # x0: imagenet-normalized image 0 [B, C, H, W]
        # x1: imagenet-normalized image 1 [B, C, H, W]

        # ensure inputs in
        x0 = self.prepare_inputs(x0)
        x1 = self.prepare_inputs(x1)
        with torch.no_grad():
            _, flow_up = self.model(x0, x1, iters=20, test_mode=True)

        if return_magnitude:
            flow_magnitude = flow_up.norm(p=2, dim=1)  # [B, H, W]
            return flow_up, flow_magnitude

        return flow_up

    def viz(self, flow):
        flow_rgb = flow_viz.flow_to_image(flow[0].permute(1,2,0).cpu().numpy())
        return flow_rgb


class FlowSampleFilter(nn.Module):
    """
    Filter out flow samples based on a list of filter methods
    - patch_magnitude: filter out samples if the flow magnitude at the selected patch is too small
    - flow_area: filter out samples if there is a large motion across the scene
    - num_corners: filter out samples if the flow covers more than 1 corner of the image

    @param filter_methods: list of filter methods
    @param flow_magnitude_threshold: threshold for patch_magnitude filter
    @param flow_area_threshold: threshold for flow_area filter
    @param num_corners_threshold: threshold for num_corners filter

    """

    ALL_FILTERS = ['patch_magnitude', 'flow_area', 'num_corners']
    def __init__(self,
                filter_methods=ALL_FILTERS,
                flow_magnitude_threshold=5.0,
                flow_area_threshold=0.75,
                num_corners_threshold=2):
        super(FlowSampleFilter, self).__init__()

        # filtering methods and hyperparameters
        self.filter_methods = filter_methods
        self.flow_magnitude_threshold = flow_magnitude_threshold
        self.flow_area_threshold = flow_area_threshold
        self.num_corners_threshold = num_corners_threshold

    def __repr__(self):
        return ("filtering by %s\nusing flow_magnitude_threshold %0.1f\n" + \
            "using flow_area_threshold %0.2f\n" + \
            "using num_corners_threshold %d") % \
            (self.filter_methods, self.flow_magnitude_threshold,
             self.flow_area_threshold, self.num_corners_threshold)

    def compute_flow_magnitude(self, flow_samples, move_pos=None):
        """
        Compute the flow magnitude over the entire image and at the selected patches of the second frame

        @param flow_samples: [B, num_samples, 2, H, W]
        @param move_pos: [B, num_samples, 2]

        @return:
            flow_mag: flow magnitude of shape [B, H, W, num_samples]
            flow_mag_down: downsized flow magnitude of shape, [B, h, w, num_samples] where h = H // patch_size
            patch_flow_mag: average flow magnitude at selected patches in frame 2 [B, num_samples]
            active_second: indication of which patch in frame 2 is active, [B, num_samples, hw] (or None)
        """

        # Compute flow magnitude map
        flow_mag = flow_samples.norm(dim=2, p=2)  # [B, num_samples, H, W]

        # Compute flow magnitude at the moving patches, if move_pos is not None
        if move_pos is not None:
            B, num_samples, _, H, W = flow_samples.shape
            assert move_pos.shape[1] == num_samples, (move_pos.shape, num_samples)

            patch_flow_mag = sample_embedding(
                embedding=flow_mag.flatten(0, 1).unsqueeze(-1), # [B * num_samples, H, W, 1]
                pos=move_pos.flatten(0, 1).unsqueeze(1), # [B * num_samples, 1, 2]
                mode='bilinear'
            ) # [B * num_samples, 1, 1]

            patch_flow_mag = patch_flow_mag.view(B, num_samples)  # [B, num_samples]

            return flow_mag, patch_flow_mag
        else:
            return flow_mag

    def filter_by_patch_magnitude(self, patch_flow_mag):
        """
        Filter out samples if the flow magnitude at the selected patch is too small (< self.flow_magnitude_threshold)
        @param patch_flow_mag: average flow magnitude at the selected patch of shape (B, S)
        @return: filter mask of shape (B, S), 1 for samples to be filtered out, 0 otherwise
        """
        assert self.flow_magnitude_threshold is not None
        return patch_flow_mag < self.flow_magnitude_threshold

    def filter_by_flow_area(self, flow_mag):
        """
        Filter out samples if there is a large motion across the scene (> self.flow_area_threshold)
        @param flow_mag: flow magnitude of shape (B, S, H, W)
        @return: boolean mask of shape (B, S), 1 for samples to be filtered out, 0 otherwise
        """
        assert self.flow_magnitude_threshold is not None
        assert self.flow_area_threshold is not None
        _, _, H, W = flow_mag.shape
        flow_area = (flow_mag > self.flow_magnitude_threshold).flatten(2, 3).sum(-1) / (H * W) # [B, num_samples]
        return flow_area > self.flow_area_threshold

    def filter_by_num_corners(self, flow_mag):
        """
        Filter out samples if the flow covers more than 1 corner of the image
        @param flow_mag: flow magnitude of shape (B, S, H, W)
        @return: boolean mask of shape (B, S), 1 for samples to be filtered out, 0 otherwise
        """
        assert self.flow_magnitude_threshold is not None
        # Binarize flow magnitude map
        flow_mag_binary = (flow_mag > self.flow_magnitude_threshold).float()

        # Get the four corners of the flow magnitude map
        top_l, top_r, bottom_l, bottom_r = flow_mag_binary[:, :, 0:1, 0:1], flow_mag_binary[:, :, 0:1,-1:], \
                                           flow_mag_binary[:, :, -1:, 0:1], flow_mag_binary[:, :, -1:,-1:]
        top_l = top_l.flatten(2, 3).max(-1)[0]
        top_r = top_r.flatten(2, 3).max(-1)[0]
        bottom_l = bottom_l.flatten(2, 3).max(-1)[0]
        bottom_r = bottom_r.flatten(2, 3).max(-1)[0]

        # Add up the 4 corners
        num_corners = top_l + top_r + bottom_l + bottom_r

        return num_corners >= self.num_corners_threshold


    def forward(self, flow_samples, move_pos):
        """
        @param flow_samples: flow samples, [B, num_samples, 2, H, W]
        @param move_pos: position of move patches, [B, num_samples, 2]

        @return: filtered flow samples of shape [B, 2, H, W, num_samples]. Flow sampled being filtered is set to zero
        @return: filtered mask of shape [B, num_samples]. 1 means the sample is filtered out
        """
        B, num_samples, _, H, W = flow_samples.shape

        # Compute flow magnitude maps and the average flow magnitude at active patches
        flow_mag, patch_flow_mag = self.compute_flow_magnitude(flow_samples, move_pos)

        # Initialize the filter mask, 0 for keeping the sample, 1 for filtering out the sample
        filter_mask = torch.zeros(B, num_samples).to(flow_samples.device).bool()

        # Iterate through all filter methods and update the filter mask
        for method in self.filter_methods:
            if method == 'patch_magnitude':
                _filter_mask = self.filter_by_patch_magnitude(patch_flow_mag)  # [B, num_samples]
            elif method == 'flow_area':
                 _filter_mask = self.filter_by_flow_area(flow_mag)  # [B, num_samples]
            elif method == 'num_corners':
                _filter_mask = self.filter_by_num_corners(flow_mag)  # [B, num_samples]
            else:
                raise ValueError(f'Filter method must be one of {self.filter_methods}, but got {method}')

            filter_mask = filter_mask | _filter_mask # [B, num_samples]

        # Apply the filter mask and set the rejected flow_samples to be zero
        filter_mask = filter_mask.view(B, num_samples, 1, 1, 1).contiguous() # [B,num_samples,1,1,1]
        filter_mask = filter_mask.expand_as(flow_samples) # [B, num_samples, 2, H, W]
        flow_samples[filter_mask] = 0. # [B, num_samples, 2, H, W]

        return flow_samples.contiguous(), filter_mask # [B, num_samples, 2, H, W], [B, num_samples]
