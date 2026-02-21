# Homework Assignments for the Mila Robot Learning Course

Youtube [link](https://youtube.com/playlist?list=PLMe2pHxzxHp-UJ1jd-uuGSGK7P7Phtm-f&si=xZ2EyWJgbAzoOycQ)

Link to details on the course is [here](https://fracturedplane.notion.site/Robot-Learning-IFT6163-Scaling-Learning-for-Real-World-Agents-Apprentissage-robotique-Apprentiss-2c4214857276801f8a22ce2c80a044a2?pvs=74).

## Install

'''module load cudatoolkit/11.8 miniconda/3'''

conda create -n roble python=3.10
conda activate roble
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install torch==2.4.0

Tips on using hydra
https://www.sscardapane.it/tutorials/hydra-tutorial/

### Install MilaTools

pip install milatools==0.1.14 decorator==4.4.2 moviepy==1.0.3

## Dataset

https://rail-berkeley.github.io/bridgedata/

## Install SimpleEnv

Prerequisites:

    CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
    An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

Clone this repo:

```
git clone https://github.com/milarobotlearningcourse/SimplerEnv --recurse-submodules
```

Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):

```
pip install numpy==1.24.4
```

Install ManiSkill2 real-to-sim environments and their dependencies:

```
cd SimplerEnv/ManiSkill2_real2sim
pip install -e .
```

Install this package:

```
cd {this_repo}
pip install -e .
```

conda install conda-forge::vulkan-tools conda-forge::vulkan-headers

## Install LIBERO

pip install cmake==3.24.3

https://github.com/montrealrobotics/LIBERO

## Running the code

Basic example to train the GRP over the bridge dataset 

```
python mini-grp/mini-grp.py
```

## Docker

The Dockerfile contains the setup to run the provided environment in a containter for better portabilty.

```
docker build -t gberseth/roble:latest .
```
Run in docker (settings for low memory use)

```
docker run --gpus=all gberseth/roble:latest python mini-grp/mini_grp.py dataset.buffer_size=1000 trim=1000 n_embd=256 batch_size=64 dataset.encode_with_t5=false data_shuffel_interval=10 eval_interval=10 dataset.num_episodes=1 dataset.chunk_size=1
```

### Mila Code

```
mila code --cluster mila playground/mini-grp/ --alloc --gres gpu:1 --mem=32G --cpus-per-gpu=6 --partition unkillable

```

### License

MIT
