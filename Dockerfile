FROM continuumio/miniconda3:latest

WORKDIR ./src

# TODO: Add support for installing using GPU support
RUN apt-get update && apt-get install build-essential -y
RUN conda install gxx_linux-64
RUN conda install -c anaconda pyyaml psycopg2
RUN conda install -c conda-forge uvicorn

COPY pylurch ./pylurch/pylurch
COPY setup.py ./pylurch

RUN pip install ./pylurch
RUN rm -rf ./pylurch

COPY example ./example

CMD uvicorn example.app:init_app --port 8080 --host 0.0.0.0 --factory --workers 3