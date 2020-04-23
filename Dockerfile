FROM ubuntu:18.04
RUN apt-get update \
   && apt-get install -y python3.7 \
   && apt-get install -y python3.7-dev \
   && apt-get install -y python3-pip \
   && apt-get install -y software-properties-common \
   && add-apt-repository ppa:sumo/stable \
   && apt-get update \
   && apt-get install -y sumo=1.5.0+dfsg1-5 sumo-tools=1.5.0+dfsg1-5 sumo-doc
ENV SUMO_HOME "/usr/share/sumo"
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
