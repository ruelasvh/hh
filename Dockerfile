# base image
#FROM python:3.9-slim-bookworm
FROM htcondor/mini:10.0-ubu20.04 

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
RUN python3 -m pip install --upgrade pip
RUN python3 --version
RUN python3 -m pip --version

# copy and install package
COPY . /hh
RUN python3 -m pip install --no-cache-dir -e /hh
