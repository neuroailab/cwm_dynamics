import numpy as np
from physion_evaluator.feature_extract_interface import PhysionFeatureExtractor
from physion_evaluator.utils import DataAugmentationForVideoMAE

from torch.functional import F

from cwm.inference.flow.flow_utils import get_occ_masks

from cwm.model.model_factory import model_factory
import torch

def load_predictor(
        model_func_,
        load_path_,
        **kwargs):
    predictor = model_func_(**kwargs).eval().requires_grad_(False)

    did_load = predictor.load_state_dict(
        torch.load(load_path_, map_location=torch.device("cpu"))['model'])
    predictor._predictor_load_path = load_path_
    print(did_load, load_path_)
    return predictor


class CWM(PhysionFeatureExtractor):
    def __init__(self, model_name, aggregate_embeddings=False):
        super().__init__()

        self.model = model_factory.load_model(model_name).cuda().half()

        self.num_frames = self.model.num_frames

        self.timestamps = np.arange(self.num_frames)

        ps = (224 // self.model.patch_size[1]) ** 2

        self.bool_masked_pos = np.zeros([ps * self.num_frames])
        self.bool_masked_pos[ps * (self.num_frames - 1):] = 1

        self.ps = ps

        self.aggregate_embeddings = aggregate_embeddings

    def transform(self):

        return DataAugmentationForVideoMAE(
            imagenet_normalize=True,
            rescale_size=224,
        ), 150, 4

    def fwd(self, videos):
        bool_masked_pos = torch.tensor(self.bool_masked_pos).to(videos.device).unsqueeze(0).bool()
        bool_masked_pos = torch.cat([bool_masked_pos] * videos.shape[0])
        x_encoded = self.model(videos.half(), bool_masked_pos, forward_full=True,
                                  return_features=True)
        return x_encoded

    def extract_features(self, videos, for_flow=False):
        '''
        videos: [B, T, C, H, W], T is 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''

        videos = videos.transpose(1, 2)

        all_features = []

        # repeat the last frame of the video
        videos = torch.cat([videos, videos[:, :, -1:]], dim=2)

        for x in range(0, 4, self.num_frames - 1):
            vid = videos[:, :, x:x + self.num_frames, :, :]
            all_features.append(self.fwd(vid))
            if self.aggregate_embeddings:
                feats = all_features[-1].mean(dim=1, keepdim=True)
                all_features[-1] = feats
                # feats = feats.view(feats.shape[0], -1, self.model.num_patches_per_frame, feats.shape[-1])
                # feats = feats.mean(dim=2)
                # all_features[-1] = feats

        x_encoded = torch.cat(all_features, dim=1)

        return x_encoded


class CWM_Keypoints(PhysionFeatureExtractor):
    def __init__(self, model_name):
        super().__init__()

        self.model = model_factory.load_model(model_name).cuda().half()

        self.frames = [[0, 1, 2], [1, 2, 3]]

        self.num_frames = self.model.num_frames

        self.ps = (224 // self.model.patch_size[1]) ** 2

        self.bool_masked_pos = np.zeros([self.ps * self.num_frames])
        self.bool_masked_pos[self.ps * (self.num_frames - 1):] = 1

        self.frame_gap = 150

        self.num_frames_dataset = 4

        self.res = 224


    def transform(self):

        return DataAugmentationForVideoMAE(
            imagenet_normalize=True,
            rescale_size=self.res,
        ), self.frame_gap, self.num_frames_dataset

    def fwd(self, videos):
        bool_masked_pos = torch.tensor(self.bool_masked_pos).to(videos.device).unsqueeze(0).bool()
        bool_masked_pos = torch.cat([bool_masked_pos] * videos.shape[0])
        _, x_encoded = self.model(videos.half(), bool_masked_pos, forward_full=True, return_features=True)
        return x_encoded

    def extract_features(self, videos):
        '''

        :param videos: [B, T, C, H, W], T is 4 and videos are normalized with imagenet norm
        :returns: [B, T, D] extracted features
        '''

        videos = videos.transpose(1, 2)

        all_features = []

        for x, arr in enumerate(self.frames):

            #use the downsampled videos for keypoints
            vid = videos[:, :, arr, :, :].half()
            frame0 = vid[:, :, 0]
            frame1 = vid[:, :, 1]
            frame2 = vid[:, :, 2]

            #extract features from the video frames frame0 and frame1 and include features at keypoint regions of frame2
            mask, choices, err_array, k_feat, keypoint_recon = self.model.get_keypoints(frame0, frame1, frame2,  10, 1)

            #reshape the features to [batch size, num_features]
            k_feat = k_feat.view(k_feat.shape[0], -1)

            all_features.append(k_feat)

        x_encoded = torch.cat(all_features, dim=1)

        return x_encoded


class CWM_KeypointsFlow(PhysionFeatureExtractor):
    def __init__(self, model_name):
        super().__init__()

        self.model = model_factory.load_model(model_name).cuda().half()

        self.frames = [[0, 3, 6], [3, 6, 9], [6, 9, 9]]

        self.num_frames = self.model.num_frames

        self.timestamps = np.arange(self.num_frames)

        self.ps = (224 // self.model.patch_size[1]) ** 2

        self.bool_masked_pos = np.zeros([self.ps * self.num_frames])
        self.bool_masked_pos[self.ps * (self.num_frames - 1):] = 1

        self.frame_gap = 50

        self.num_frames_dataset = 9

        self.res = 512

    def transform(self):

        return DataAugmentationForVideoMAE(
            imagenet_normalize=True,
            rescale_size=self.res,
        ), self.frame_gap, self.num_frames_dataset

    def fwd(self, videos):
        bool_masked_pos = torch.tensor(self.bool_masked_pos).to(videos.device).unsqueeze(0).bool()
        bool_masked_pos = torch.cat([bool_masked_pos] * videos.shape[0])
        _, x_encoded = self.model(videos.half(), bool_masked_pos, forward_full=True,
                                  return_features=True)
        return x_encoded

    def get_forward_flow(self, videos):

        fid = 6

        forward_flow = self.model.get_flow_cost_volume_method(videos[:, :, fid], videos[:, :, fid + 1], conditioning_img=videos[:, :, fid + 2])

        backward_flow = self.model.get_flow_cost_volume_method(videos[:, :, fid + 1], videos[:, :, fid], conditioning_img=videos[:, :, fid - 1])

        occlusion_mask = get_occ_masks(forward_flow, backward_flow)[0]

        forward_flow = forward_flow * occlusion_mask

        forward_flow = torch.stack([forward_flow, forward_flow, forward_flow], dim=1)

        forward_flow = forward_flow.to(videos.device)

        forward_flow = F.interpolate(forward_flow, size=(2, 224, 224), mode='nearest')

        return forward_flow

    def extract_features(self, videos):
        '''
        :param videos: [B, T, C, H, W] videos normalized with imagenet norm
        :return: [B, T, D] extracted features
        Note:
        For efficiency, the optical flow is computed and added for a single frame (300ms) as we found this to be sufficient
        for capturing temporal dynamics in our experiments. This approach can be extended to multiple frames if needed,
        depending on the complexity of the task.
        :return: [B, T, D] extracted features
        '''

        #resize to 224 to get keypoints and features
        videos_downsampled = F.interpolate(videos.flatten(0, 1), size=(224, 224), mode='bilinear', align_corners=False)
        videos_downsampled = videos_downsampled.view(videos.shape[0], videos.shape[1], videos.shape[2], 224, 224)

        #for computing flow at higher resolution
        videos_ = F.interpolate(videos.flatten(0, 1), size=(1024, 1024), mode='bilinear', align_corners=False)
        videos = videos_.view(videos.shape[0], videos.shape[1], videos.shape[2], 1024, 1024)

        videos = videos.transpose(1, 2).half()
        videos_downsampled = videos_downsampled.transpose(1, 2).half()

        # Get the forward flow for the frame at 300ms
        forward_flow = self.get_forward_flow(videos)

        # Verify that there are no nans forward flow
        assert not torch.isnan(forward_flow).any(), "Forward flow is nan"

        all_features = []

        for x, arr in enumerate(self.frames):

            #use the downsampled videos for keypoints
            vid = videos_downsampled[:, :, arr, :, :]
            frame0 = vid[:, :, 0]
            frame1 = vid[:, :, 1]
            frame2 = vid[:, :, 2]

            #extract features from the video frames frame0 and frame1 and include features at keypoint regions of frame2
            mask, choices, err_array, k_feat, keypoint_recon = self.model.get_keypoints(frame0, frame1, frame2,  10, 1)

            #for the last set of frames only use features at keypoint regions of frame2
            if (x == 2):
                k_feat = k_feat[:, -10:, :]

            #reshape the features to [batch size, num_features]
            k_feat = k_feat.view(k_feat.shape[0], -1)

            choices_image_resolution = choices * self.model.patch_size[1]

            # At 300ms, add optical flow patches at the detected keypoint locations
            # For the first frame (x == 0)
            if x == 0:
                # Extract the optical flow information from the forward flow matrix for the second channel (index 2)
                flow_keyp = forward_flow[:, 2]

                # Initialize a result tensor to store the flow patches
                # Tensor shape: [batch_size, 8x8 patch (flattened to 64) * 2 channels, 10 keypoints]
                flow = torch.zeros(vid.shape[0], 8 * 8 * 2, 10).to(videos.device)

                # Patch size shift (since 8x8 patches are being extracted)
                shift = 8

                # Loop over each element in the batch to process individual video frames
                for b in range(flow_keyp.size(0)):
                    # Extract the x and y coordinates of the keypoint locations for this batch element
                    x_indices = choices_image_resolution[b, :, 0]
                    y_indices = choices_image_resolution[b, :, 1]

                    # For each keypoint (10 total keypoints in this case)
                    for ind in range(10):
                        # Extract the 8x8 patch of optical flow at each keypoint's (x, y) location
                        # Flatten the patch and assign it to the corresponding slice in the result tensor
                        flow[b, :, ind] = flow_keyp[b, :, y_indices[ind]:y_indices[ind] + shift,
                                          x_indices[ind]:x_indices[ind] + shift].flatten()

                # Reshape the flow tensor for easier concatenation (flatten across all patches)
                flow = flow.view(flow.shape[0], -1)

                # Concatenate the extracted optical flow features with the existing feature tensor (k_feat)
                k_feat = torch.cat([k_feat, flow], dim=1)

            all_features.append(k_feat)

        x_encoded = torch.cat(all_features, dim=1)

        return x_encoded


class CWM_base_8x8_3frame(CWM):
    def __init__(self,):
        super().__init__('vitb_8x8patch_3frames')

class CWM_base_8x8_3frame_mean_embed(CWM):
    def __init__(self,):
        super().__init__('vitb_8x8patch_3frames', aggregate_embeddings=True)

# CWM (keypoints only)
class CWM_base_8x8_3frame_keypoints(CWM_Keypoints):
    def __init__(self,):
        super().__init__('vitb_8x8patch_3frames')


# CWM (keypoints + flow)
class CWM_base_8x8_3frame_keypoints_flow(CWM_KeypointsFlow):
    def __init__(self,):
        super().__init__('vitb_8x8patch_3frames')

