#!/bin/bash

docker run -it --rm -v "$(pwd)":/app py3:opencv \
  /bin/bash -c "pipenv install && pipenv run python person_detect.py"

