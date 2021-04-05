FROM python:3

WORKDIR /app

# install opencv lib deps, tools
RUN apt-get update && apt-get install -y python3-opencv curl

# install required pips
RUN python -m pip install pipenv
