FROM continuumio/anaconda3:2021.05

# to use GPUs
ENV NVIDIA_VISIBLE_DEVICES all

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev

COPY . /workdir
WORKDIR /workdir

# create virtual environment and activate
RUN conda env create -f tf-keras.yml

RUN echo "conda activate tf-keras" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

EXPOSE 8888