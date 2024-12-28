# Use an official NVIDIA CUDA image as a base for PyTorch with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Ensure the system is up-to-date and install required packages
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

# Create and activate the conda environment
RUN conda create -n geoldm python=3.10.13 -y && \
    conda init bash && \
    echo "conda activate geoldm" >> ~/.bashrc

# Initialize conda in bash
# RUN conda init bash

# RUN conda init

# Install required packages in the conda environment
# RUN /bin/bash -c "source ~/.bashrc && \
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate geoldm && \
    conda install -c conda-forge rdkit biopython openbabel -y && \
    conda install pathtools==0.1.2 -y && \
    pip install imageio numpy==1.23.3 scipy tqdm wandb==0.13.4 msgpack rdkit \
    matplotlib==3.5.2 matplotlib-inline==0.1.6 chardet periodictable ipykernel jupyter notebook \
    prettytable seaborn scikit-learn==1.5.1 gdown && \
    pip install gradio==5.9 plotly==5.24 huggingface && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

# Create and activate the conda environment
RUN conda create -n mgltools-python2 python=2.7 -y && \
    conda init bash

# RUN /bin/bash -c "source ~/.bashrc && \
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate mgltools-python2 && \
    conda install -c bioconda mgltools -y"

# Clone the repository
RUN git clone https://github.com/PIEthonista/GeoLDM-edit /usr/src/geoldm-edit

# Change directory, make the file executable, and return to the base directory
RUN cd /usr/src/geoldm-edit/analysis/qvina && \
    chmod +x qvina2.1

# Clone the huggingface weights (you might need to configure git-lfs for large files)
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    git clone https://huggingface.co/PIEthonista/control-geoldm-deployment /usr/src/geoldm-edit/deployment/models/controlnet

# Set working directory
WORKDIR /usr/src/geoldm-edit

# Set up the container to start Jupyter Notebook server
# CMD ["/bin/bash"]
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate geoldm"]