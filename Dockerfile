# Base container that includes all dependencies but not the actual repo
# Updated from templates in the [softlearning (SAC) library](https://github.com/rail-berkeley/softlearning)

# FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM dsalvat1/cudagl:12.3.1-runtime-ubuntu22.04
# FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04 as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND="noninteractive"
# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Set environment variables
ENV MINICONDA_HOME /opt/conda
ENV PATH=$MINICONDA_HOME/bin:$PATH

# --- CRITICAL FIX START ---
# 1. Set NVIDIA capabilities immediately so runtime hooks know what to mount
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

# 2. Add system library paths to LD_LIBRARY_PATH
# Conda environments often isolate themselves. We must force them to look
# in /usr/lib/x86_64-linux-gnu where the NVIDIA Container Toolkit mounts the driver .so files.
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:$LD_LIBRARY_PATH
# --- CRITICAL FIX END ---

# Install necessary build tools and download Miniconda
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
# Always check repo.anaconda.com/miniconda for the latest installer link
# Use -b for batch mode (no prompts) and -p for the installation prefix
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p $MINICONDA_HOME && \
    rm miniconda.sh

# Accept Conda Terms of Service for default channels
# This is the crucial part to fix the CondaToSNonInteractiveError
RUN conda config --set plugins.auto_accept_tos yes && \
    conda init bash && \
    conda clean --all -f -y

RUN conda create --name roble python=3.10 pip
RUN echo "source activate roble" >> ~/.bashrc
## Make it so you can install things to the correct version of pip
ENV PATH /opt/conda/envs/roble/bin:$PATH
RUN source activate roble

# Set the working directory for your application
WORKDIR /playground

## Install the requirements for your learning code.
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -c "import libero"

## Install pytorch and cuda
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## Install simulators simpleEnv
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg libx264-dev
# RUN pip install cmake==3.24.3
RUN conda install -c conda-forge cmake
RUN git clone https://github.com/milarobotlearningcourse/SimplerEnv --recurse-submodules
## Change directory to SimplerEnv and install ManiSkill2 and ManiSkill2_real2sim
# RUN cd SimplerEnv/ManiSkill2
# RUN cd SimplerEnv/ManiSkill2_real2sim
RUN pip install -e ./SimplerEnv/ManiSkill2_real2sim
# RUN cd ../
RUN pip install -e ./SimplerEnv
# RUN cd ../
RUN apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools
RUN conda install conda-forge::vulkan-tools conda-forge::vulkan-headers

# 2. MANUALLY Generate the NVIDIA Vulkan ICD (The critical fix)
# This tells Vulkan to use the NVIDIA driver instead of looking for a display
RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version" : "1.3.0" } }' > /etc/vulkan/icd.d/nvidia_icd.json

# 3. Setup EGL (Required for headless SAPIEN/PyRender)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# 4. Set Environment Variables permanently in the image
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
# This prevents SAPIEN from trying to open a GUI window
ENV SAP_NO_GUI=1
ENV DISPLAY=:0

## Install Libero
# RUN pip install cmake==3.24.3
RUN apt-get update && apt-get install -y --no-install-recommends \
cmake \
libglvnd-dev \
libgl1-mesa-dev \
libegl1-mesa-dev \
libgles2-mesa-dev \
libgbm-dev \
build-essential \
&& rm -rf /var/lib/apt/lists/*
# RUN git clone https://github.com/montrealrobotics/LIBERO.git
# COPY --link ./LIBERO /playground/LIBERO
# RUN pip install -r ./LIBERO/requirements.txt
# RUN pip install -e ./LIBERO 
## Check that libero is installed and easy to load


## Check the file were copied
RUN ls
COPY --link . /playground
# RUN pip install -e ./LIBERO 
# RUN git clone https://github.com/montrealrobotics/LIBERO.git
# # COPY --link ./LIBERO /playground/LIBERO
# RUN pip install -r ./LIBERO/requirements.txt
# RUN pip install -e ./LIBERO 

ENTRYPOINT [ "python" ]
