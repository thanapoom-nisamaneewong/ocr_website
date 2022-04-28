FROM python:3.9 as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y 
RUN apt-get install libleptonica-dev -y 
RUN apt-get install tesseract-ocr -y
RUN apt-get install tesseract-ocr-eng -y
RUN apt-get install tesseract-ocr-tha -y
RUN apt-get install libtesseract-dev -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

WORKDIR /home/ocr-web

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

# production image.
FROM base as production

WORKDIR /production

COPY . .
# COPY ocr-web/static ./
# COPY ocr-web/templates ./