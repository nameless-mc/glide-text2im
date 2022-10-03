FROM nvidia/cuda:11.0.3-devel-ubuntu18.04

RUN mkdir /app
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip git

COPY script/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

CMD ["python3", "-B", "main.py"]
