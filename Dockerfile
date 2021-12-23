# poetry docker:  https://zenn.dev/yacchi21/articles/b210985793b90f
# multi-stage build: https://future-architect.github.io/articles/20200513/
ARG PYTHON_ENV_BUILDER=python:3.8.11-buster
ARG PYTHON_ENV=python:3.8.11-slim-buster

# build stage
FROM $PYTHON_ENV_BUILDER as build

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 libgl1 make gcc \
        ninja-build libglib2.0-0 libxrender-dev libgl1-mesa-glx 

WORKDIR /project
COPY ./pyproject.toml ./poetry.lock /project

RUN python -m pip install --upgrade pip && \
    python -m pip install poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-dev && \
    rm -rf ~/.cache

# development stage
FROM $PYTHON_ENV as development

COPY --from=build /usr/lib /usr/lib
RUN apt-get update && \
    apt-get install -y git vim

WORKDIR /project
COPY ./pyproject.toml ./poetry.lock /project
COPY --from=build /project/.venv /project/.venv
ENV PATH=/project/.venv/bin:$PATH

RUN python -m pip install --upgrade pip && \
    python -m pip install poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install

# m1 mac上でonnx/onnxruntimeはインストールできないので，別途インストール
RUN python -m pip install onnx==1.10.2 onnxruntime==1.8.0

# production stage
FROM $PYTHON_ENV as production

COPY --from=build /usr/lib /usr/lib
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /project/.venv /project/.venv
ENV PATH=/project/.venv/bin:$PATH

CMD ["/bin/bash"]

