# Base image.
# FROM nvcr.io/nvidia/pytorch:24.01-py3
# FROM nvcr.io/nvidia/cuda:12.5.1-runtime-ubuntu24.04
FROM nvcr.io/nvidia/cuda:11.6.1-runtime-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# Install packages.
RUN apt-get update -y
RUN apt-get install -y make
RUN apt-get install -y lzma
RUN apt-get install -y liblzma-dev
RUN apt-get install -y gcc 
RUN apt-get install -y zlib1g-dev bzip2 libbz2-dev
RUN apt-get install -y libreadline8 
RUN apt-get install -y libreadline-dev
RUN apt-get install -y sqlite3 libsqlite3-dev
RUN apt-get install -y openssl libssl-dev build-essential 
RUN apt-get install -y git curl wget
RUN apt-get install -y vim
RUN apt-get install -y sudo
RUN apt-get install -y libffi-dev
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

RUN apt-get install -y lsb-release gnupg
RUN apt-get install -y python3.10 python3-pip
RUN apt-get install -y byobu

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Env variables.
ARG USERNAME
ENV USERNAME=$USERNAME
ARG UID
ENV UID=$UID
ARG GID
ENV GID=$GID
ARG ORIGINAL_DATA_LOC
ENV ORIGINAL_DATA_LOC=$ORIGINAL_DATA_LOC

# Prepare user and directory.
RUN addgroup -gid $GID $USERNAME
RUN adduser $USERNAME --uid $UID --gid $GID
RUN usermod -aG sudo $USERNAME
RUN echo "$USERNAME:$USERNAME" | chpasswd

RUN mkdir -p $ORIGINAL_DATA_LOC
RUN chmod 777 $ORIGINAL_DATA_LOC
WORKDIR /home/$USERNAME
USER $USERNAME
ENV HOME /home/$USERNAME

# Pyenv.
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN source ~/.bashrc

# Install python.
RUN pyenv install 3.9.6
RUN pyenv global 3.9.6
RUN source ~/.bashrc

# Install libraries
RUN pip install --upgrade pip
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install numpy==1.24.2
RUN pip install pandas==1.5.3
RUN pip install matplotlib==3.9.1
RUN pip install optuna==3.6.1
RUN pip install scikit-learn==1.5.1
RUN pip install scipy==1.13.1
RUN pip install timm==0.9.12
RUN pip install tqdm
RUN pip install pyYaml
