FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_md

COPY . .

ARG VERSION=0.0.0
ENV VERSION=${VERSION}
RUN pip install -e .

RUN chmod +x ./runserver.sh

ENTRYPOINT ./runserver.sh
