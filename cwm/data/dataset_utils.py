from torchvision import transforms
from cwm.data.transforms import *
from cwm.data.dataset import ContextAndTargetVideoDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from cwm.data.masking_generator import RotatedTableMaskingGenerator

class DataAugmentationForVideoMAE(object):
    def __init__(self, augmentation_type, input_size, augmentation_scales):

        transform_list = []

        self.scale = GroupScale(input_size)
        transform_list.append(self.scale)

        if augmentation_type == 'multiscale':
            self.train_augmentation = GroupMultiScaleCrop(input_size, list(augmentation_scales))
        elif augmentation_type == 'center':
            self.train_augmentation = GroupCenterCrop(input_size)

        transform_list.extend([self.train_augmentation, Stack(roll=False), ToTorchFormatTensor(div=True)])

        # Normalize input images
        normalize = GroupNormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        transform_list.append(normalize)

        self.transform = transforms.Compose(transform_list)

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


def build_pretraining_dataset(args):

    dataset_list = []
    data_transform = DataAugmentationForVideoMAE(args.augmentation_type, args.input_size, args.augmentation_scales)

    mask_generator = RotatedTableMaskingGenerator(
        input_size=args.mask_input_size,
        mask_ratio=args.mask_ratio,
        tube_length=args.tubelet_size,
        batch_size=1,
        mask_type=args.mask_type
    )

    for data_path in [args.data_path] if args.data_path_list is None else args.data_path_list:
        dataset = ContextAndTargetVideoDataset(
            root=None,
            setting=data_path,
            video_ext='mp4',
            context_length=args.context_frames,
            target_length=args.target_frames,
            step_units=args.temporal_units,
            new_step=args.sampling_rate,
            context_target_gap=args.context_target_gap,
            transform=data_transform,
            randomize_interframes=False,
            channels_first=True,
            temporal_jitter=False,
            train=True,
            mask_generator=mask_generator,
        )
        dataset_list.append(dataset)
    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset
