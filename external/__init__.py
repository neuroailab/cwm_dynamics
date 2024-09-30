import os
import subprocess
import torch
import sys

def setup_raft():
    print('RAFT is not installed. Auto-install RAFT to ~/.cache/torch/RAFT')
    # Store the current working directory
    initial_directory = os.getcwd()
    os.makedirs(os.path.join(os.environ['HOME'], '.cache/torch/'), exist_ok=True)
    os.chdir(os.path.join(os.environ['HOME'], '.cache/torch/'))
    subprocess.run(['git', 'clone', 'https://github.com/princeton-vl/RAFT.git'], check=True)
    os.chdir('RAFT')
    subprocess.run(['./download_models.sh'], check=True)
    # Change back to the initial directory
    os.chdir(initial_directory)

def setup_cutler():
    print('CUTLER is not installed. Auto-install CUTLER to ~/.cache/torch/CUTLER')
    # Store the current working directory
    initial_directory = os.getcwd()
    os.makedirs(os.path.join(os.environ['HOME'], '.cache/torch/'), exist_ok=True)
    os.chdir(os.path.join(os.environ['HOME'], '.cache/torch/'))
    subprocess.run(['git', 'clone', '--recursive', 'https://github.com/facebookresearch/CutLER.git'], check=True)
    subprocess.run(['pip', 'install', 'git+https://github.com/lucasb-eyer/pydensecrf.git'], check=True)
    # subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/detectron2.git'], check=True)
    # os.chdir('detectron2')
    # subprocess.run(['pip', 'install', '-e', '.'], check=True)

    url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(url)
    del state_dict

    # Change back to the initial directory
    os.chdir(initial_directory)

if not os.path.isdir(os.path.join(os.environ['HOME'], '.cache/torch/', 'RAFT')):
    setup_raft()

if not os.path.isdir(os.path.join(os.environ['HOME'], '.cache/torch/', 'CutLER')):
    setup_cutler()
