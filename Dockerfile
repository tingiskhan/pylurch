FROM continuumio/miniconda3:latest

WORKDIR ./src

# TODO: Add support for installing using GPU support
RUN apt-get update && apt-get install build-essential -y
RUN conda install gxx_linux-64
RUN conda install -c anaconda pyyaml

COPY ml_server ./ml_server/ml_server
COPY setup.py ./ml_server/

RUN pip install ./ml_server/.

ENTRYPOINT ["gunicorn", "-b :8080", "ml_server.app:init_app()"]