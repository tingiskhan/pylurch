FROM continuumio/miniconda3:latest

WORKDIR ./src

# TODO: Add support for installing using GPU support
RUN apt-get update && apt-get install build-essential -y
RUN conda install gxx_linux-64
RUN conda install -c anaconda pyyaml gunicorn psycopg2

RUN conda install pytorch torchvision cpuonly -c pytorch
RUN conda install -c conda-forge pytorch-lightning

COPY pylurch ./pylurch/pylurch
COPY setup.py ./pylurch

RUN pip install ./pylurch

COPY example ./example

ENTRYPOINT ["gunicorn", "-b :8080", "-w 3", "example.app:init_app()"]