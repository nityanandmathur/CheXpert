# Nvidia 11.6 cuda version
FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

RUN apt-get update && apt-get -y upgrade
RUN export DEBIAN_FRONTEND=noninteractive   
RUN apt-get -y install git

# Latest nodejs for jupyterlab and its extenstions
RUN apt-get -y install curl
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install -y nodejs

# Latest Miniconda installation
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
    
RUN conda install -c conda-forge jupyterlab nodejs -y
RUN conda --version
RUN conda init bash