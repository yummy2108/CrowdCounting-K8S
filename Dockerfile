FROM python:3.6.8
WORKDIR /app 
COPY . .
COPY requirements.txt /app 
RUN python3 -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r ./requirements.txt 
COPY app.py /app 
CMD ["python", "app.py"] 