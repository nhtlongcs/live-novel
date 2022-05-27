FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Package version control

ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=11.3
ARG PYTORCH_VERSION=1.10
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch


# Setup workdir and non-root user

ARG USERNAME=dreamer
WORKDIR /home/$USERNAME/workspace/

# https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update &&\
    apt-get install -y --no-install-recommends curl git sudo &&\
    useradd --create-home --shell /bin/bash $USERNAME &&\
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME &&\
    chmod 0440 /etc/sudoers.d/$USERNAME &&\
    rm -rf /var/lib/apt/lists/*

RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get -qq update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    gcc \
    tmux \
    libjpeg-dev \
    unzip bzip2 ffmpeg libsm6 libxext6 \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*


# Install repo dependencies 
COPY ./ $WORKDIR
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/

SHELL ["/bin/bash","-c"]
RUN python -m pip install -r requirements.txt

USER $USERNAME
# RUN conda init bash 
# ENTRYPOINT ["sh", "run.sh"]