FROM continuumio/miniconda3:latest

WORKDIR ./src

# TODO: Add support for installing using GPU support
RUN apt-get update && apt-get install build-essential -y
RUN conda install gxx_linux-64
RUN conda install -c anaconda pyyaml gunicorn

RUN pip install git+https://github.com/tingiskhan/ml-server.git
RUN conda install -c conda-forge flask-httpauth flask-bcrypt

COPY example ./example

ENTRYPOINT ["gunicorn", "-b :8080", "example.app:init_app()"]