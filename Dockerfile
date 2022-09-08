FROM --platform=linux/amd64 nvidia/cuda:11.6.0-devel-ubuntu20.04 

RUN echo "downloading basic packages for installation"
RUN apt-get update
RUN apt-get install -y tmux wget curl
RUN apt-get install -y libstdc++6 gcc

# set up conda
RUN mkdir -p src
RUN cd src

# install conda
RUN wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b -p src/conda
RUN rm Miniconda3-latest-Linux-x86_64.sh
RUN . "src/conda/etc/profile.d/conda.sh"
ENV PATH="src/conda/condabin:${PATH}"
RUN conda create --name tankbind-conda python=3.7 -y

# checking installation of tools
RUN gcc --version
RUN nvcc --version

# Switch to the new environment:
SHELL ["conda", "run", "-n", "tankbind-conda", "/bin/bash", "-c"] 
RUN conda update -n base conda -y
RUN conda install pytorch cudatoolkit=11.3 -c pytorch
RUN conda install torchdrug=0.1.2 pyg biopython nglview jupyterlab -c milagraph -c conda-forge -c pytorch -c pyg
RUN pip install torchmetrics tqdm mlcrate pyarrow

RUN echo ready