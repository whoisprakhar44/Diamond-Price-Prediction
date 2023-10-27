FROM python:3.9-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN apt-get update && pip install -r requirements.txt

EXPOSE 5000

CMD [ "python3", "app.py" ] 
