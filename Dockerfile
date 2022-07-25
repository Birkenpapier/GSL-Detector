FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  qt5-default -y \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      && rm -rf /var/lib/apt/lists/*

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

ENV QT_X11_NO_MITSHM=1
ENV TF_CPP_MIN_LOG_LEVEL=3

COPY . .

CMD [ "python3", "3_Real_Time.py"]
