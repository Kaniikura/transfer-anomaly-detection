FROM rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04-py3.8

# install ffmpeg for sk-video
RUN apt-get update && \
    apt-get -y install software-properties-common
RUN add-apt-repository ppa:savoury1/ffmpeg4; \
    apt-get update && \
    apt-get -y install ffmpeg

# Install PyTorch and other packages
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate rapids \
    && pip install -U torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/cu111/torch_stable.html \
    && pip install efficientnet_pytorch==0.7.0 pytorch_lightning==1.1.1 wandb==0.10.12 albumentations==0.5.2 \
    seaborn==0.11.0 omegaconf==2.0.5 scikit-video==1.1.11
WORKDIR /work/share
