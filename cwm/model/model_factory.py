"""
Model Factory for loading a checkpoint from gcloud, then initializing the model and the configuration
"""
import os
import requests
import torch
import tqdm

from cwm.model import model_pretrain

GCLOUD_BUCKET_NAME = "stanford_neuroai_models"
GCLOUD_URL_NAME = "https://storage.googleapis.com/stanford_neuroai_models"
CACHE_PATH = f"{os.getenv('CACHE')}/stanford_neuroai_models" if os.getenv('CACHE') is not None else ".cache/stanford_neuroai_models"
_model_catalogue ={
    "vitb_8x8patch_3frames": {
        "path": "cwm/3frame_cwm_8x8.pth",
        "init_fn": model_pretrain.vitb_8x8patch_3frames,
    },
    "vitb_4x4patch_2frames": {
        "path": "cwm/2frame_cwm_4x4.pth",
        "init_fn": model_pretrain.vitb_4x4patch_2frames,
    },

    "vitb_8x8patch_2frames": {
        "path": "cwm/2frame_cwm_8x8.pth",
        "init_fn": model_pretrain.vitb_8x8patch_2frames,
    },

    "vitb_8x8patch_2frames_learnable_pos_embed": {
        "path": "cwm/2frame_cwm_mask_token.pth",
        "init_fn": model_pretrain.vitb_8x8patch_2frames_learnable_pos_embed,
    },
    "vitl_8x8patch_3frames": {
        "path": "cwm/3frame_cwm_8x8_large.pth",
        "init_fn": model_pretrain.vitl_8x8patch_3frames,
    },

}


class ModelFactory:

    def __init__(self, bucket_name: str = GCLOUD_BUCKET_NAME):
        self.bucket_name = bucket_name

    def get_catalog(self):
        """
        Get the list of available models
        """
        # Initialize the storage client
        return _model_catalogue.keys()

    def load_model(self, model_name: str, force_download=False):
        """
        Load the model given the name

        Args:
        model_name: str
            Name of the model to load
        force_download: bool (optional)
            Whether to force the download of the freshest weights from gcloud

        Returns:
        model: torch.nn.Module
            Model initialized from the checkpoint
        """
        # Find cache dir, use a directory inside of it as the checkpoint path
        checkpoint_path = os.path.join(CACHE_PATH, _model_catalogue[model_name]["path"])
        # Make checkpoint directory if it does not exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        # Construct gcloud url
        gcloud_url = os.path.join(GCLOUD_URL_NAME, _model_catalogue[model_name]['path'])
        # Download the model from google cloud using requests (with tqdm timer)
        response = requests.get(gcloud_url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        # If force_download is true or the model has not yet been downloaded, grab it from gcloud
        if force_download or not os.path.exists(checkpoint_path):
            progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

            print(f"Saving model to cache: {CACHE_PATH}")
            with open(checkpoint_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        # Initialize the model and the configuration from the checkpoint
        print("checkpoint_path", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Initialize the model from the specified initialization function
        model = _model_catalogue[model_name]["init_fn"]()


        # Load the model from the checkpoint
        model.load_state_dict(ckpt['model'], strict=True)
        print('Model loaded successfully')

        return model

model_factory = ModelFactory()