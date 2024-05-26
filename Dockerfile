FROM bentoml/model-server:0.11.0-py310
MAINTAINER ersilia

RUN pip install rdkit-pypi==2022.9.5
RUN pip install protobuf==3.18.3
RUN pip install tensorboardx==2.0
RUN pip install chemprop==1.5.2
RUN pip install tensorboard==2.11.0
RUN pip install ScaffoldGraph==1.1.2
RUN pip install networkx==3.3

# Clone the repository and checkout the desired commit
RUN git clone --branch v4.2.6 --single-branch https://github.com/MolecularAI/REINVENT4

# Install the package using pip
RUN pip install ./reinvent4 --extra-index-url=https://pypi.anaconda.org/OpenEye/simple --extra-index-url=https://download.pytorch.org/whl/cu113

WORKDIR /repo
COPY . /repo