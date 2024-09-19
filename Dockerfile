# base image
FROM python:3.9-slim-bookworm

# local and envs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive

# add some packages
RUN apt-get update && apt-get install -y git git-lfs h5utils wget vim build-essential

# update python pip
RUN python3 -m pip install --upgrade pip
RUN python3 --version
RUN python3 -m pip --version

# copy source code and install package
COPY . /opt/hh
RUN python3 -m pip install --no-cache-dir -e /opt/hh

# create working directory
WORKDIR /home
