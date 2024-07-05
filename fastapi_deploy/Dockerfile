FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/fastapi_deploy
COPY my-creds.json /code/fastapi_deploy
