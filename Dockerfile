FROM python:3.11.5-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    libopenslide-dev \
    openslide-tools \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libssl-dev \
    libncurses5-dev \
    libsqlite3-dev \
    libreadline-dev \
    libtk8.6 \
    libvips \
    libvips-dev \
    libgdm-dev \
    libdb4o-cil-dev \
    libpcap-dev \
    && pip install --no-cache-dir openslide-python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*