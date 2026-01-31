# SPARCED Devcontainer
# Based on docker/jupyter.Dockerfile but modified for VS Code devcontainer usage
# This provides a full development environment rather than just a Jupyter notebook server

FROM ubuntu:20.04

# install system dependencies
RUN apt-get update -qq && apt-get install -qq -y \
    curl \
    git \
    python3-dev \
    python3-pip \
    libhdf5-serial-dev \
    libatlas-base-dev \
    swig \
    rsync \
    && ln -s /usr/bin/python3 python

# set working directory
WORKDIR /workspace

# copy requirements file
COPY requirements.txt /tmp/requirements.txt

# install python dependencies from requirements.txt
# upgrade pip and setuptools first to fix compatibility with amici
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# install jupyter (optional, can be started manually when needed)
RUN pip3 install --no-cache-dir jupyter

# remove tini and jupyter entrypoint - we want a bash shell for development
# keep container running with sleep infinity
CMD ["sleep", "infinity"]
