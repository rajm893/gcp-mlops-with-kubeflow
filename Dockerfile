FROM gcr.io/deeplearning-platform-release/base-cpu
WORKDIR /
COPY training_pipeline.py /
COPY requirements.txt /
COPY ./src/ /src
RUN pip install --upgrade pip && pip install -r requirements.txt
