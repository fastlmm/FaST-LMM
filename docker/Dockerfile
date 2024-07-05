# Use 16.04 as it is the current linux default in the travis build 
# https://github.com/fastlmm/FaST-LMM/blob/master/.travis.yml
FROM ubuntu:16.04

# gcc and g++ are critical dependencies for pysnptools
RUN apt-get update && \
  apt-get install -y wget gcc g++ && \
  rm -rf /var/lib/apt/lists/*

# Install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    chown -R $USERNAME:$USERNAME /opt/conda

# Initialize environment
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n fastlmm python=3.7
RUN conda init bash
RUN echo "conda activate fastlmm" >> ~/.bashrc
ENV PATH /opt/conda/envs/fastlmm/bin:$PATH
RUN conda install -n fastlmm "mkl==2019.4" "scipy>=1.1.0" "numpy>=1.11.3" jupyter

# While not ideal, pysnptools/fastlmm should be installed separately via pip
# See: https://github.com/fastlmm/FaST-LMM/issues/6#issuecomment-598458316
RUN /opt/conda/envs/fastlmm/bin/pip install pysnptools --no-use-pep517
RUN /opt/conda/envs/fastlmm/bin/pip install fastlmm

# Initialize some working directory for bind mounts
RUN mkdir /work
WORKDIR /work
