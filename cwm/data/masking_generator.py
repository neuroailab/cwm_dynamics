import numpy as np
import torch

def get_tubes(masks_per_frame, tube_length):
    rp = torch.randperm(len(masks_per_frame))
    masks_per_frame = masks_per_frame[rp]

    tubes = [masks_per_frame]
    for x in range(tube_length - 1):
        masks_per_frame = masks_per_frame.clone()
        rp = torch.randperm(len(masks_per_frame))
        masks_per_frame = masks_per_frame[rp]
        tubes.append(masks_per_frame)

    tubes = torch.vstack(tubes)

    return tubes

class RotatedTableMaskingGenerator:
    def __init__(self,
                 input_size,
                 mask_ratio,
                 tube_length,
                 batch_size,
                 mask_type='rotated_table',
                 seed=None,
                 randomize_num_visible=False):

        self.batch_size = batch_size

        self.mask_ratio = mask_ratio
        self.tube_length = tube_length

        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame

        self.seed = seed
        self.randomize_num_visible = randomize_num_visible

        self.mask_type = mask_type

    def __repr__(self):
        repr_str = "Inverted Table Mask: total patches {}, tube length {}, randomize num visible? {}, seed {}".format(
            self.total_patches, self.tube_length, self.randomize_num_visible, self.seed
        )
        return repr_str

    def __call__(self, m=None):

        if self.mask_type == 'rotated_table_magvit':
            self.mask_ratio = np.random.uniform(low=0.0, high=1)
            self.mask_ratio = np.cos(self.mask_ratio * np.pi / 2)
        elif self.mask_type == 'rotated_table_maskvit':
            self.mask_ratio = np.random.uniform(low=0.5, high=1)

        all_masks = []
        for b in range(self.batch_size):

            self.num_masks_per_frame = max(0, int(self.mask_ratio * self.num_patches_per_frame))
            self.total_masks = self.tube_length * self.num_masks_per_frame

            num_masks = self.num_masks_per_frame

            if self.randomize_num_visible:
                assert "Randomize num visible Not implemented"
                num_masks = self.rng.randint(low=num_masks, high=(self.num_patches_per_frame + 1))

            if self.mask_ratio == 0:
                mask_per_frame = torch.hstack([
                torch.zeros(self.num_patches_per_frame - num_masks),
            ])
            else:
                mask_per_frame = torch.hstack([
                    torch.zeros(self.num_patches_per_frame - num_masks),
                    torch.ones(num_masks),
                ])

            tubes = get_tubes(mask_per_frame, self.tube_length)
            top = torch.zeros(self.height * self.width).to(tubes.dtype)

            top = torch.tile(top, (self.frames - self.tube_length, 1))
            mask = torch.cat([top, tubes])
            mask = mask.flatten()
            all_masks.append(mask)
        return torch.stack(all_masks)
