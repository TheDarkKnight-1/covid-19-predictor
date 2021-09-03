FROM tensorflow/tensorflow:2.3.0

ADD requirements.txt /
RUN pip install -r /requirements.txt &&\
    apt-get update -y &&\
    apt-get install -y p7zip-full
    
ADD . /app

WORKDIR /app/models
RUN 7z e datty.7z.001

WORKDIR /app

EXPOSE 5000
CMD [ "flask" , "run"]
