# Use official NVIDIA CUDA image as base
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Environment variables
ENV GITHUB_REPO=https://github.com/PIEthonista/GeoLDM-edit
ENV GITHUB_LOCAL_PATH=/usr/src/geoldm-edit
ENV HUGGINGFACE_REPO=https://huggingface.co/PIEthonista/control-geoldm-deployment
ENV MODEL_WEIGHTS_PATH=${GITHUB_LOCAL_PATH}/deployment/models/controlnet
ENV CONDA_ENV_MAIN=geoldm
ENV CONDA_ENV_MGLTOOLS=mgltools-python2


# Ensure system is up-to-date
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda update -y conda

# Modify .bashrc to enable conda in non-interactive shells
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Create main conda env and set bash terminal to activate it by default
RUN conda create -n ${CONDA_ENV_MAIN} python=3.10.13 -y && \
    conda init bash && \
    echo "conda activate ${CONDA_ENV_MAIN}" >> ~/.bashrc

# Install required packages in the main conda environment
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_MAIN} && \
    conda install -c conda-forge rdkit biopython openbabel -y && \
    conda install pathtools==0.1.2 -y && \
    pip install imageio numpy==1.23.3 scipy tqdm wandb==0.13.4 msgpack rdkit networkx \
        matplotlib==3.5.2 matplotlib-inline==0.1.6 chardet periodictable ipykernel jupyter notebook \
        prettytable seaborn scikit-learn==1.5.1 gdown gradio==5.9 plotly==5.24 huggingface && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113"

# Create helper conda env
RUN conda create -n ${CONDA_ENV_MGLTOOLS} python=2.7 -y && \
    conda init bash

# Install required packages in the helper conda environment
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_MGLTOOLS} && \
    conda install -c bioconda mgltools -y"

# Clone the code repository
RUN git clone ${GITHUB_REPO} ${GITHUB_LOCAL_PATH}

# Modify QuickVina2.1 binaries' permissions to executable
RUN cd ${GITHUB_LOCAL_PATH}/analysis/qvina && \
    chmod +x qvina2.1

# Clone model weights from huggingface
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    git clone ${HUGGINGFACE_REPO} ${MODEL_WEIGHTS_PATH}

# Set working directory
WORKDIR ${GITHUB_LOCAL_PATH}

# Set up the container to start with bash terminal
CMD ["/bin/bash"]