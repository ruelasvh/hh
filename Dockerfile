# base image
FROM python:3.9-slim-bookworm

# local and envs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /home

# add some packages
RUN apt-get update && apt-get install -y git git-lfs h5utils wget vim build-essential

# update python pip
RUN python -m pip install --upgrade pip
RUN python --version
RUN python -m pip --version

# copy and install package
COPY . /hh
RUN python -m pip install --no-cache-dir --no-deps -e /hh
