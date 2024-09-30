import matplotlib.pyplot as plt
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
import kornia

def imshow(ims, ax=None, t=0, vmin=None, vmax=None, title=None, cmap=None, fontsize=20):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    with torch.no_grad():
        im = ims[t].float().cpu().numpy().transpose((1, 2, 0))
    if (vmin is not None) and (vmax is not None):
        im = ax.imshow(im, vmin=vmin, vmax=vmax, cmap=(cmap or 'viridis'))
    else:
        im = ax.imshow(im)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    return (im, ax)


def get_video(video_name, num_frames=2, delta_time=4, frame=None):
    decord_vr = VideoReader(video_name, num_threads=1, ctx=cpu(0))
    max_end_ind = len(decord_vr) - num_frames * delta_time - 1
    start_frame = frame if frame is not None else rng.randint(1, max_end_ind)
    print("fps", decord_vr.get_avg_fps())
    print("start frame = %d" % start_frame)
    frame_id_list = list(range(start_frame, start_frame + num_frames * delta_time, delta_time))
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = [Image.fromarray(video_data[t]).convert('RGB') for t, _ in enumerate(frame_id_list)]
    return (torch.stack([transforms.ToTensor()(im) for im in video_data], 0), start_frame)


class FlowToRgb(object):

    def __init__(self, max_speed=1.0, from_image_coordinates=True, from_sampling_grid=False):
        self.max_speed = max_speed
        self.from_image_coordinates = from_image_coordinates
        self.from_sampling_grid = from_sampling_grid

    def __call__(self, flow):
        assert flow.size(-3) == 2, flow.shape
        if self.from_sampling_grid:
            flow_x, flow_y = torch.split(flow, [1, 1], dim=-3)
            flow_y = -flow_y
        elif not self.from_image_coordinates:
            flow_x, flow_y = torch.split(flow, [1, 1], dim=-3)
        else:
            flow_h, flow_w = torch.split(flow, [1,1], dim=-3)
            flow_x, flow_y = [flow_w, -flow_h]


        # print("flow_x", flow_x[0, :, 0, 0], flow_y[0, :, 0, 0])
        angle = torch.atan2(flow_y, flow_x) # in radians from -pi to pi
        speed = torch.sqrt(flow_x**2 + flow_y**2) / self.max_speed

        # print("angle", angle[0, :, 0, 0] * 180 / np.pi)

        hue = torch.fmod(angle, torch.tensor(2 * np.pi))
        sat = torch.ones_like(hue)
        val = speed

        hsv = torch.cat([hue, sat, val], -3)
        rgb = kornia.color.hsv_to_rgb(hsv)
        return rgb
